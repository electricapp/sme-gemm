// Runtime SME capability probe via sysctl. Returns 1 if the named
// `hw.optional.arm.*` flag is present and nonzero, else 0.
#include <stddef.h>
#include <stdint.h>
#include <sys/sysctl.h>

int sme_sysctl_flag(const char *name);

int sme_sysctl_flag(const char *name) {
    int64_t v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) {
        return 0;
    }
    return v != 0;
}
