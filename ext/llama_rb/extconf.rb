require 'mkmf'

extension_name = 'llama_rb'
dir_config(extension_name)

# keep standard at C11 and C++11
$CFLAGS   << " -O3 -DNDEBUG -std=c11 -fPIC"
$CXXFLAGS << " -O3 -DNDEBUG -std=c++11 -fPIC"

# warnings
$CFLAGS   << " -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function"
$CXXFLAGS << " -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function"

uname_s = `uname -s`.strip rescue ""
uname_p = `uname -p`.strip rescue ""
uname_m = `uname -m`.strip rescue ""

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
if uname_s == "Darwin" && uname_p != "arm"
  sysctl_m = `sysctl -n hw.optional.arm64 2>/dev/null`.strip

  if sysctl_m == "1"
    warn "Your arch is announced as x86_64, but it seems to actually be ARM64. "\
      "Not fixing this can lead to bad performance. " \
      "For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)"
  end
end

# OS specific
# TODO: support Windows
if %w[Linux Darwin FreeBSD NetBSD OpenBSD Haiku].include?(uname_s)
  $CFLAGS   << " -pthread"
  $CXXFLAGS << " -pthread"
end

# Architecture specific
# TODO: these flags probably need to be tweaked on some architectures
if uname_m.include?("x86_64") || uname_m.include?("i686")
  if uname_s == "Darwin"
    cpu_features = `sysctl machdep.cpu.features`.strip
    cpu_leaf7_features = `sysctl machdep.cpu.leaf7_features`.strip

    $CFLAGS << " -mf16c" if cpu_features.include?("F16C")
    $CFLAGS << " -mfma"  if cpu_features.include?("FMA")
    $CFLAGS << " -mavx"  if cpu_features.include?("AVX1.0")
    $CFLAGS << " -mavx2" if cpu_leaf7_features.include?("AVX2")
  elsif uname_s == "Linux"
    cpuinfo = File.read("/proc/cpuinfo")

    $CFLAGS << " -mavx"        if cpuinfo =~ /\bavx\b/
    $CFLAGS << " -mavx2"       if cpuinfo =~ /\bavx2\b/
    $CFLAGS << " -mfma"        if cpuinfo =~ /\bfma\b/
    $CFLAGS << " -mf16c"       if cpuinfo =~ /\bf16c\b/
    $CFLAGS << " -msse3"       if cpuinfo =~ /\bsse3\b/
    $CFLAGS << " -mavx512f"    if cpuinfo =~ /\bavx512f\b/
    $CFLAGS << " -mavx512bw"   if cpuinfo =~ /\bavx512bw\b/
    $CFLAGS << " -mavx512dq"   if cpuinfo =~ /\bavx512dq\b/
    $CFLAGS << " -mavx512vl"   if cpuinfo =~ /\bavx512vl\b/
    $CFLAGS << " -mavx512cd"   if cpuinfo =~ /\bavx512cd\b/
    $CFLAGS << " -mavx512er"   if cpuinfo =~ /\bavx512er\b/
    $CFLAGS << " -mavx512ifma" if cpuinfo =~ /\bavx512ifma\b/
    $CFLAGS << " -mavx512pf"   if cpuinfo =~ /\bavx512pf\b/
  elsif uname_s == "Haiku"
    sysinfo_cpu = `sysinfo -cpu`.strip

    $CFLAGS << " -mavx"  if sysinfo_cpu =~ /\bAVX\b/
    $CFLAGS << " -mavx2" if sysinfo_cpu =~ /\bAVX2\b/
    $CFLAGS << " -mfma"  if sysinfo_cpu =~ /\bFMA\b/
    $CFLAGS << " -mf16c" if sysinfo_cpu =~ /\bF16C\b/
  else
    $CFLAGS << " -mfma -mf16c -mavx -mavx2"
  end
end

if uname_m =~ /ppc64.+/
  cpuinfo = File.read("/proc/cpuinfo")

  if cpuinfo =~ /\bPOWER9\b/
    $CFLAGS   << " -mcpu=power9"
    $CXXFLAGS << " -mcpu=power9"
  end

  # Require c++23's std::byteswap for big-endian support.
  CXXFLAGS << " -std=c++23 -DGGML_BIG_ENDIAN" if uname_m == "ppc64"
end

if ENV.fetch("LLAMA_NO_ACCELERATE", "false") == "false"
  puts "got past first if"
  # Mac M1 - include Accelerate framework.
  # `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
  if uname_s == "Darwin"
    $CFLAGS  << " -DGGML_USE_ACCELERATE"
    $LDFLAGS << " -framework Accelerate"
  end
end

if ENV.fetch("LLAMA_OPENBLAS", "false") == "true"
  $CFLAGS  << " -DGGML_USE_OPENBLAS -I/usr/local/include/openblas"
  have_library("openblas")
end

if ENV.fetch("LLAMA_GPROF", "true") == "true"
  $CFLAGS   << " -pg"
  $CXXFLAGS << " -pg"
end

if uname_m =~ /aarch64.+/
  $CFLAGS   << " -mcpu=native"
  $CXXFLAGS << " -mcpu=native"
end

if uname_m =~ /armv6.+/
  # Raspberry Pi 1, 2, 3
  $CFLAGS << " -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access"
end

if uname_m =~ /armv7.+/
  # Raspberry Pi 4
  $CFLAGS << " -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations"
end

if uname_m =~ /armv8.+/
  # Raspberry Pi 4
  $CFLAGS << " -mfp16-format=ieee -mno-unaligned-access"
end

include_paths = [
  File.expand_path('./', __dir__),
  File.expand_path('./llama-cpp/', __dir__),
  File.expand_path('./llama-cpp/examples/', __dir__),
]

$srcs = []

include_paths.each do |include_path|
  $INCFLAGS << " -I#{include_path}"
  $VPATH << include_path

  Dir.glob("#{include_path}/*.{c,cpp}").each do |path|
    $srcs << path
  end
end

# $CFLAGS << ' -O3 -pthread -mf16c -mfma -mavx -mavx2 -DGGML_USE_ACCELERATE'
# $CXXFLAGS << ' -O3 -pthread'
# $LDFLAGS << ' -framework Accelerate'

puts "llama.rb build info:"
puts "UNAME_S:  #{uname_s}"
puts "UNAME_P:  #{uname_p}"
puts "UNAME_M:  #{uname_m}"
puts "CFLAGS:   #{$CFLAGS}"
puts "CXXFLAGS: #{$CXXFLAGS}"
puts "LDFLAGS:  #{$LDFLAGS}"

create_makefile(extension_name)
