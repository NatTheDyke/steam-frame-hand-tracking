// slamgrab: publish the Steam Frame's lower tracking (SLAM) camera frames to /dev/shm/mrhand-slam.
//
// XRService keeps every SLAM camera frame in udmabuf dma-bufs it already owns.  This reader borrows
// those buffers read-only (pidfd_getfd + mmap PROT_READ) -- it never writes to them, never pauses or
// attaches to XRService, and works while the passthrough cameras are paused (games, room view off).
//
// Ring layout (XRService, SteamVR 2.19.2 on the headset; see HANDOFF / docs):
//   per frame one image buffer (1769472 B; luma 1056x1024 packed, pitch 1056) and one metadata buffer
//   (151552 B) allocated back to back, so their dma-buf inodes are consecutive.  16 such pairs for the
//   RIGHT camera, then 16 for the LEFT camera.  Unpaired 1769472 B buffers are compositor copies.
//   metadata u64 at offset 0 = capture time in system-counter ticks << 8 (19.2 MHz counter).
//   Each camera alternates a scene exposure and a controller-LED exposure 8.38 ms later (period 33.3 ms);
//   only scene frames are published.
//
// Output: same layout as ptgrab's /dev/shm/mrhand-pt (MrhandPtHeader, capture/handcam_proto.h), 2 cams,
// slot 0 = slam_right, slot 1 = slam_left, raw sensor orientation.  slot.tDetectNs = CAPTURE time
// (metadata timestamp converted to CLOCK_MONOTONIC), not detection time.
//
// Privilege: pidfd_getfd needs CAP_SYS_PTRACE (Yama ptrace_scope=1).  Install with
//   sudo setcap cap_sys_ptrace+p build/slamgrab
// The capability is permitted but not effective; it is raised only around pidfd_getfd.
//
// usage: slamgrab [stream] [--max-fps N]     slamgrab check   (tearing / timing probe, no shm)
#include <dirent.h>
#include <fcntl.h>
#include <linux/capability.h>
#include <linux/dma-buf.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "handcam_proto.h"

static const uint64_t IMG_SIZE = 1769472, META_SIZE = 151552;
static const uint32_t W = 1056, H = 1024, RING = 16;
static const char *SHM = "/dev/shm/mrhand-slam";

static uint64_t monoNs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static uint64_t cntvct() {
    uint64_t v;
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static uint64_t cntfrq() {
    uint64_t v;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

// ---- capabilities: raise / drop CAP_SYS_PTRACE in the effective set (no libcap dependency)
static bool setPtraceEffective(bool on) {
    __user_cap_header_struct h{_LINUX_CAPABILITY_VERSION_3, 0};
    __user_cap_data_struct d[2]{};
    if (syscall(SYS_capget, &h, d) != 0) return false;
    if (on) d[0].effective |= 1u << CAP_SYS_PTRACE;
    else d[0].effective &= ~(1u << CAP_SYS_PTRACE);
    return syscall(SYS_capset, &h, d) == 0;
}

static int findXRService() {
    DIR *d = opendir("/proc");
    if (!d) return -1;
    int found = -1;
    while (dirent *e = readdir(d)) {
        int pid = atoi(e->d_name);
        if (pid <= 0) continue;
        char path[64], exe[512];
        snprintf(path, sizeof path, "/proc/%d/exe", pid);
        ssize_t n = readlink(path, exe, sizeof exe - 1);
        if (n <= 0) continue;
        exe[n] = 0;
        if (strstr(exe, "/drivers/cv/bin/linuxarm64/XRService")) { found = pid; break; }
    }
    closedir(d);
    return found;
}

struct Buf {
    uint64_t ino = 0, size = 0;
    int remoteFd = -1, fd = -1;
    const uint8_t *map = nullptr;
};
struct Pair { Buf img, meta; };

static void dmaSync(int fd, uint64_t flags) {
    dma_buf_sync s{flags};
    ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
}
static uint64_t metaTicks(const Pair &p) {
    dmaSync(p.meta.fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
    uint64_t v;
    memcpy(&v, p.meta.map, 8);
    dmaSync(p.meta.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
    return v >> 8;
}

// Enumerate XRService's dma-bufs (unprivileged: same uid), pair image+meta by consecutive inode,
// then borrow the 32 pairs with pidfd_getfd.  Returns false (with a reason) if the layout is not
// the expected one -- never guesses.
static bool attach(int pid, int &pidfd, std::vector<Pair> &pairs, std::string &why) {
    pairs.clear();
    std::vector<Buf> bufs;
    char dirp[64];
    snprintf(dirp, sizeof dirp, "/proc/%d/fd", pid);
    DIR *d = opendir(dirp);
    if (!d) { why = "cannot list XRService fds"; return false; }
    while (dirent *e = readdir(d)) {
        int n = atoi(e->d_name);
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char p[128], l[256];
        snprintf(p, sizeof p, "/proc/%d/fd/%d", pid, n);
        ssize_t k = readlink(p, l, sizeof l - 1);
        if (k <= 0) continue;
        l[k] = 0;
        if (strncmp(l, "/dmabuf", 7) != 0) continue;
        snprintf(p, sizeof p, "/proc/%d/fdinfo/%d", pid, n);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        Buf b;
        b.remoteFd = n;
        char line[256];
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "size:", 5)) b.size = strtoull(line + 5, nullptr, 10);
            if (!strncmp(line, "ino:", 4)) b.ino = strtoull(line + 4, nullptr, 10);
        }
        fclose(f);
        if (b.size == IMG_SIZE || b.size == META_SIZE) bufs.push_back(b);
    }
    closedir(d);
    std::sort(bufs.begin(), bufs.end(), [](const Buf &a, const Buf &b) { return a.ino < b.ino; });
    for (size_t i = 0; i + 1 < bufs.size(); i++)
        if (bufs[i].size == IMG_SIZE && bufs[i + 1].size == META_SIZE && bufs[i + 1].ino == bufs[i].ino + 1) {
            pairs.push_back({bufs[i], bufs[i + 1]});
            i++;
        }
    if (pairs.size() != 2 * RING) {
        why = "found " + std::to_string(pairs.size()) + " image+meta pairs, expected 32 (unknown XRService layout?)";
        pairs.clear();
        return false;
    }
    pidfd = (int)syscall(SYS_pidfd_open, pid, 0);
    if (pidfd < 0) { why = std::string("pidfd_open: ") + strerror(errno); return false; }
    setPtraceEffective(true);
    for (auto &p : pairs)
        for (Buf *b : {&p.img, &p.meta}) {
            b->fd = (int)syscall(SYS_pidfd_getfd, pidfd, b->remoteFd, 0);
            if (b->fd < 0) {
                int err = errno;
                setPtraceEffective(false);
                why = std::string("pidfd_getfd: ") + strerror(err) +
                      (err == EPERM ? " (run: sudo setcap cap_sys_ptrace+p slamgrab)" : "");
                return false;
            }
        }
    setPtraceEffective(false);
    for (auto &p : pairs) {
        p.img.map = (const uint8_t *)mmap(nullptr, W * H, PROT_READ, MAP_SHARED, p.img.fd, 0);
        p.meta.map = (const uint8_t *)mmap(nullptr, 4096, PROT_READ, MAP_SHARED, p.meta.fd, 0);
        if (p.img.map == MAP_FAILED || p.meta.map == MAP_FAILED) { why = "mmap failed"; return false; }
    }
    return true;
}

static void detach(int &pidfd, std::vector<Pair> &pairs) {
    for (auto &p : pairs) {
        if (p.img.map && p.img.map != MAP_FAILED) munmap((void *)p.img.map, W * H);
        if (p.meta.map && p.meta.map != MAP_FAILED) munmap((void *)p.meta.map, 4096);
        if (p.img.fd >= 0) close(p.img.fd);
        if (p.meta.fd >= 0) close(p.meta.fd);
    }
    pairs.clear();
    if (pidfd >= 0) close(pidfd);
    pidfd = -1;
}

static bool processGone(int pidfd) {
    pollfd p{pidfd, POLLIN, 0};
    return poll(&p, 1, 0) > 0;
}

static volatile sig_atomic_t gStop = 0;

int main(int argc, char **argv) {
    bool check = argc > 1 && !strcmp(argv[1], "check");
    int maxFps = 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--max-fps") && i + 1 < argc) maxFps = atoi(argv[++i]);
    signal(SIGINT, [](int) { gStop = 1; });
    signal(SIGTERM, [](int) { gStop = 1; });
    setPtraceEffective(false);

    const double frq = (double)cntfrq();
    // counter -> CLOCK_MONOTONIC offset (ns); minimum over samples bounds the read latency
    auto offsetNs = []() {
        int64_t best = INT64_MAX;
        for (int i = 0; i < 64; i++) {
            uint64_t m0 = monoNs(), c = cntvct(), m1 = monoNs();
            int64_t o = (int64_t)llround((double)c * 1e9 / (double)cntfrq()) - (int64_t)((m0 + m1) / 2);
            best = std::min(best, o);
        }
        return best;
    };

    MrhandPtHeader *hdr = nullptr;
    uint8_t *slots = nullptr;
    const size_t frame = (size_t)W * H;
    if (!check) {
        int fd = open(SHM, O_RDWR | O_CREAT, 0600);
        size_t total = sizeof(MrhandPtHeader) + 2 * frame;
        if (fd < 0 || ftruncate(fd, total) != 0) { perror("shm"); return 1; }
        uint8_t *base = (uint8_t *)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        hdr = (MrhandPtHeader *)base;
        memset(hdr, 0, sizeof *hdr);
        hdr->magic = MRHAND_PT_MAGIC;
        hdr->width = W;
        hdr->height = H;
        hdr->cams = 2;
        slots = base + sizeof(MrhandPtHeader);
    }
    printf("slamgrab %s: counter %.3f MHz, output %s (slot0 = slam_right, slot1 = slam_left, %ux%u)\n",
           check ? "check" : "stream", frq / 1e6, check ? "-" : SHM, W, H);
    fflush(stdout);

    std::vector<Pair> pairs;
    int pidfd = -1;
    while (!gStop) {
        int pid = findXRService();
        std::string why;
        if (pid < 0 || !attach(pid, pidfd, pairs, why)) {
            static std::string lastWhy;
            std::string msg = pid < 0 ? "XRService not running" : why;
            if (msg != lastWhy) { printf("waiting: %s\n", msg.c_str()); fflush(stdout); lastWhy = msg; }
            detach(pidfd, pairs);
            sleep(2);
            continue;
        }
        printf("attached XRService %d: 32 pairs, right inodes %lu-%lu, left inodes %lu-%lu\n", pid,
               (unsigned long)pairs[0].img.ino, (unsigned long)pairs[RING - 1].meta.ino,
               (unsigned long)pairs[RING].img.ino, (unsigned long)pairs[2 * RING - 1].meta.ino);
        fflush(stdout);

        int64_t off = offsetNs();
        std::vector<uint64_t> seen(pairs.size());
        for (size_t i = 0; i < pairs.size(); i++) seen[i] = metaTicks(pairs[i]);
        uint64_t lastTicks[2] = {0, 0}, lastPub[2] = {0, 0};
        uint64_t nScene[2] = {0, 0}, nLed[2] = {0, 0}, nTorn[2] = {0, 0}, nPub[2] = {0, 0}, frames[2] = {0, 0};
        uint64_t nChangedLater = 0, nChecked = 0;
        uint64_t lastSceneTicks[2] = {0, 0}, lagNs = 5000000, lagMin = UINT64_MAX;   // lag: first guess 5 ms
        std::vector<double> ages, copyMs;
        uint64_t statT = monoNs(), offT = statT;
        while (!gStop) {
            if (processGone(pidfd)) { printf("XRService exited; re-attaching\n"); fflush(stdout); break; }
            // collect new timestamps, then handle them in capture order (an LED and the next scene frame
            // can both be new after a sleep)
            std::vector<std::pair<uint64_t, size_t>> fresh;
            for (size_t i = 0; i < pairs.size(); i++) {
                uint64_t t = metaTicks(pairs[i]);
                if (t == seen[i]) continue;
                seen[i] = t;
                fresh.push_back({t, i});
            }
            std::sort(fresh.begin(), fresh.end());
            for (auto [t, i] : fresh) {
                int cam = i < RING ? 0 : 1;
                // newest frame of this camera only (a ring slot can be seen late after a stall)
                if (t <= lastTicks[cam]) continue;
                double gapMs = lastTicks[cam] ? (t - lastTicks[cam]) * 1e3 / frq : 1e9;
                lastTicks[cam] = t;
                if (gapMs < 12.0) { nLed[cam]++; continue; }      // LED exposure 8.38 ms after scene
                nScene[cam]++;
                lastSceneTicks[cam] = t;
                uint64_t capNs = (uint64_t)llround((double)t * 1e9 / frq) - off;
                {   // capture -> timestamp-visible lag (min over a window), used to time the next wake-up
                    uint64_t seenNs = monoNs();
                    uint64_t lag = seenNs > capNs ? seenNs - capNs : 0;
                    lagMin = std::min(lagMin, lag);
                }
                const Pair &p = pairs[i];
                if (check) {
                    // tearing probe: hash a few rows now and again 4 ms later
                    auto hashRows = [&]() {
                        dmaSync(p.img.fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
                        uint64_t h = 1469598103934665603ull;
                        for (uint32_t y = 0; y < H; y += 37)
                            for (uint32_t x = 0; x < W; x += 13) h = (h ^ p.img.map[y * W + x]) * 1099511628211ull;
                        dmaSync(p.img.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
                        return h;
                    };
                    uint64_t h0 = hashRows();
                    ages.push_back((monoNs() - capNs) / 1e6);
                    timespec ts{0, 4000000};
                    nanosleep(&ts, nullptr);
                    nChecked++;
                    if (hashRows() != h0) nChangedLater++;
                    continue;
                }
                uint64_t now = monoNs();
                if (maxFps > 0 && now - lastPub[cam] < 1000000000ull / maxFps) continue;
                lastPub[cam] = now;
                uint64_t t0 = monoNs();
                MrhandPtSlot &s = hdr->slot[cam];
                __atomic_store_n(&s.seq, s.seq + 1, __ATOMIC_RELEASE);   // odd: writing
                dmaSync(p.img.fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
                memcpy(slots + cam * frame, p.img.map, frame);
                dmaSync(p.img.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
                bool torn = metaTicks(p) != t;   // slot recycled while copying
                if (torn) {
                    nTorn[cam]++;
                    // leave the previous frame's metadata; seq stays odd->even with old timestamp
                } else {
                    s.tDetectNs = capNs;
                    s.bufferId = (uint32_t)i;
                    s.frameCount = ++frames[cam];
                    nPub[cam]++;
                }
                __atomic_store_n(&s.seq, s.seq + 1, __ATOMIC_RELEASE);   // even: done
                copyMs.push_back((monoNs() - t0) / 1e6);
                ages.push_back((monoNs() - capNs) / 1e6);
                hdr->stat[cam]++;
            }
            uint64_t now = monoNs();
            if (now - offT > 10000000000ull) { off = offsetNs(); offT = now; }
            if (now - statT > 5000000000ull) {
                double dt = (now - statT) / 1e9;
                auto pct = [](std::vector<double> v, double q) {
                    if (v.empty()) return 0.0;
                    std::sort(v.begin(), v.end());
                    return v[std::min(v.size() - 1, (size_t)(q * v.size()))];
                };
                printf("R scene %.1f/s led %.1f/s pub %.1f/s torn %lu | L scene %.1f/s led %.1f/s pub %.1f/s torn %lu | "
                       "age ms med %.1f p95 %.1f | copy ms med %.2f",
                       nScene[0] / dt, nLed[0] / dt, nPub[0] / dt, (unsigned long)nTorn[0],
                       nScene[1] / dt, nLed[1] / dt, nPub[1] / dt, (unsigned long)nTorn[1],
                       pct(ages, 0.5), pct(ages, 0.95), pct(copyMs, 0.5));
                printf(" | visible lag ms %.1f", lagMin == UINT64_MAX ? 0.0 : lagMin / 1e6);
                if (check) printf(" | changed 4 ms after timestamp: %lu/%lu", (unsigned long)nChangedLater, (unsigned long)nChecked);
                printf("\n");
                fflush(stdout);
                for (int c = 0; c < 2; c++) { nScene[c] = nLed[c] = nPub[c] = nTorn[c] = 0; if (hdr) hdr->stat[c] = 0; }
                nChangedLater = nChecked = 0;
                if (lagMin != UINT64_MAX) lagNs = lagMin;
                lagMin = UINT64_MAX;
                ages.clear();
                copyMs.clear();
                statT = now;
            }
            // Frames come on a fixed 33.3 ms rhythm (scene, LED +8.38 ms).  Once both cameras' newest scene
            // frame is in, sleep until ~2 ms before the next one is due instead of polling every 1 ms
            // (polling 32 timestamps each ms cost ~19% of a core).
            uint64_t sleepNs = 1000000;
            if (!check && lastSceneTicks[0] && lastSceneTicks[0] == lastSceneTicks[1]) {
                uint64_t dueNs = (uint64_t)llround((double)lastSceneTicks[0] * 1e9 / frq) - off + 33333333ull + lagNs;
                uint64_t nowNs = monoNs();
                if (dueNs > nowNs + 3000000) sleepNs = dueNs - nowNs - 2000000;
                if (sleepNs > 40000000) sleepNs = 1000000;   // stale estimate (cameras paused): plain polling
            }
            timespec ts{0, (long)sleepNs};
            nanosleep(&ts, nullptr);
        }
        detach(pidfd, pairs);
    }
    return 0;
}
