// Runtime probes via sysctl: SME capability flags and the core count.
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/sysctl.h>

int sme_sysctl_flag(const char *name);
int64_t sme_sysctl_int(const char *name);

// The named integer sysctl (32- or 64-bit), or -1 if it does not exist.
int64_t sme_sysctl_int(const char *name) {
    int64_t v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) {
        return -1;
    }
    if (sz == sizeof(int32_t)) {
        int32_t w;
        memcpy(&w, &v, sizeof(w));
        return w;
    }
    return v;
}

// 1 if the named `hw.optional.arm.*` flag is present and set, else 0.
int sme_sysctl_flag(const char *name) {
    return sme_sysctl_int(name) > 0;
}
