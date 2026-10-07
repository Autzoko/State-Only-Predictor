# Source on a server: activates the `sopt` env and exports SOPT_DATA for that host.
for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$(conda info --base 2>/dev/null)"; do
  [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
done
conda activate sopt
case "$(hostname)" in
  nyuair)   export SOPT_DATA=$HOME/langtian/SOPT_DATA ;;
  chatsign) export SOPT_DATA=$HOME/Desktop/langtian/SOPT_DATA ;;
  *)        export SOPT_DATA=${SOPT_DATA:-/scratch/$USER/SOPT_DATA} ;;
esac
