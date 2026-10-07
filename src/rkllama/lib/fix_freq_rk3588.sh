#!/bin/sh
#
# Pin RK3588 clocks for NPU inference.
#
# rkllama_server runs this at startup when it runs as root:
#   bash fix_freq_rk3588.sh <debug: 0|1>
#
# Default: pin ONLY the NPU and the DDR (dmc) at max. These are the clocks
# RKLLM inference depends on. CPU frequency scaling, CPU idle states and the
# GPU are left as the system set them, so the CPU can idle cool and the
# kernel's thermal trips can still throttle it.
#
# Optional environment flags (1/true/yes/on to enable):
#   RKLLAMA_PIN_CPU=1       Also disable cpuidle state1 on cpu0-7 and pin the
#                           CPU policies with the userspace governor
#                           (A55 1.8 GHz, A76 2.352 GHz). Old default.
#   RKLLAMA_PIN_GPU=1       Also pin the Mali GPU at 1 GHz. Old default.
#                           rkllama does not use the GPU, so this is off.
#   RKLLAMA_UNPIN=1         Undo an earlier pin of whatever this run does not
#                           pin: re-enable cpuidle state1 and set the CPU
#                           governor to RKLLAMA_CPU_GOVERNOR (default
#                           schedutil); set the GPU governor to
#                           simple_ondemand. Use once after upgrading from the
#                           old script, instead of a reboot.
#   RKLLAMA_CPU_GOVERNOR=x  Governor RKLLAMA_UNPIN restores (default schedutil).
#   RKLLAMA_FREQ_DRY_RUN=1  Print every sysfs write instead of doing it.
#
# RKLLAMA_PIN_CPU=1 RKLLAMA_PIN_GPU=1 makes exactly the same writes, in the
# same order, as the script did before these flags existed.
#
# Note: rkllama's config loader also reads RKLLAMA_<SECTION>_<KEY> variables,
# so RKLLAMA_PIN_CPU shows up as a harmless "pin.cpu" entry in a debug config
# dump. Nothing in rkllama reads it.
#
# Nothing is restored when rkllama stops: the pins stay until reboot or until
# this script runs again with RKLLAMA_UNPIN=1.

# Check if debug mode is enabled (first argument)
DEBUG_MODE=${1:-0}

# Function for conditional echo based on debug mode
debug_echo() {
  if [ "$DEBUG_MODE" = "1" ]; then
    echo "$@"
  fi
}

# Print a sysfs file in debug mode, if it exists
debug_cat() {
  if [ "$DEBUG_MODE" = "1" ] && [ -r "$1" ]; then
    cat "$1"
  fi
}

# True if the flag's value is 1/true/yes/on
is_on() {
  case "$1" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# Write a value to a sysfs node, or print the write in dry-run mode
write_sysfs() {
  if is_on "$RKLLAMA_FREQ_DRY_RUN"; then
    echo "DRY-RUN: $1 > $2"
  else
    echo "$1" > "$2"
  fi
}

PIN_CPU=0; is_on "$RKLLAMA_PIN_CPU" && PIN_CPU=1
PIN_GPU=0; is_on "$RKLLAMA_PIN_GPU" && PIN_GPU=1
UNPIN=0; is_on "$RKLLAMA_UNPIN" && UNPIN=1
CPU_GOVERNOR=${RKLLAMA_CPU_GOVERNOR:-schedutil}

# CPU idle states: disable state1 (cpu-sleep) only when the CPU pin is on
if [ "$PIN_CPU" = "1" ]; then
  for cpu in 0 1 2 3 4 5 6 7; do
    write_sysfs 1 /sys/devices/system/cpu/cpu$cpu/cpuidle/state1/disable
  done
elif [ "$UNPIN" = "1" ]; then
  debug_echo "Re-enable CPU idle state1:"
  for cpu in 0 1 2 3 4 5 6 7; do
    write_sysfs 0 /sys/devices/system/cpu/cpu$cpu/cpuidle/state1/disable
  done
fi

# NPU frequency management (always)
debug_echo "NPU available frequencies:"
debug_cat /sys/class/devfreq/fdab0000.npu/available_frequencies
debug_echo "Fix NPU max frequency:"
write_sysfs userspace /sys/class/devfreq/fdab0000.npu/governor
write_sysfs 1000000000 /sys/class/devfreq/fdab0000.npu/userspace/set_freq
debug_cat /sys/class/devfreq/fdab0000.npu/cur_freq

# CPU frequency management (only with RKLLAMA_PIN_CPU=1)
if [ "$PIN_CPU" = "1" ]; then
  debug_echo "CPU available frequencies:"
  debug_cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
  debug_cat /sys/devices/system/cpu/cpufreq/policy4/scaling_available_frequencies
  debug_cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies
  debug_echo "Fix CPU max frequency:"
  write_sysfs userspace /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
  write_sysfs 1800000 /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
  debug_cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
  write_sysfs userspace /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
  write_sysfs 2352000 /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
  debug_cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
  write_sysfs userspace /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
  write_sysfs 2352000 /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed
  debug_cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq
elif [ "$UNPIN" = "1" ]; then
  debug_echo "Release CPU frequency to the $CPU_GOVERNOR governor:"
  for policy in 0 4 6; do
    write_sysfs "$CPU_GOVERNOR" /sys/devices/system/cpu/cpufreq/policy$policy/scaling_governor
  done
fi

# GPU frequency management (only with RKLLAMA_PIN_GPU=1)
if [ "$PIN_GPU" = "1" ]; then
  debug_echo "GPU available frequencies:"
  debug_cat /sys/class/devfreq/fb000000.gpu/available_frequencies
  debug_echo "Fix GPU max frequency:"
  write_sysfs userspace /sys/class/devfreq/fb000000.gpu/governor
  write_sysfs 1000000000 /sys/class/devfreq/fb000000.gpu/userspace/set_freq
  debug_cat /sys/class/devfreq/fb000000.gpu/cur_freq
elif [ "$UNPIN" = "1" ]; then
  debug_echo "Release GPU frequency to the simple_ondemand governor:"
  write_sysfs simple_ondemand /sys/class/devfreq/fb000000.gpu/governor
fi

# DDR frequency management (always)
debug_echo "DDR available frequencies:"
debug_cat /sys/class/devfreq/dmc/available_frequencies
debug_echo "Fix DDR max frequency:"
write_sysfs userspace /sys/class/devfreq/dmc/governor
write_sysfs 2112000000 /sys/class/devfreq/dmc/userspace/set_freq
debug_cat /sys/class/devfreq/dmc/cur_freq

# Summary line (also in debug mode, so the journal shows what was pinned)
PINNED="NPU DDR"
[ "$PIN_CPU" = "1" ] && PINNED="$PINNED CPU"
[ "$PIN_GPU" = "1" ] && PINNED="$PINNED GPU"
echo "RK3588 frequencies optimized for NPU inferencing (pinned: $PINNED)"
