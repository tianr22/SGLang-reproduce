# 小规模复现
下面是你可能用得到的脚本，你可能需要修改某些配置（例如加载的conda环境）
* run_prefill.sh/run_decode.sh：启动相应P/D instance
```bash
srun -N x -n x --gres=gpu:8 --cpus-per-task=32 ./run_prefill.sh
```
* run_lb.sh：启动load balancer
```bash
./run_lb.sh
```
* benchmark.sh：profile相应instance
```
./benchmark.sh [mode] [profile]
```
第一个参数为prefill/decode，第二个参数为true/false
* run_deep_ep.sh
```bash
srun -N x -n x --gres=gpu:8 --cpus-per-task=21 ./run_deep_ep.sh [mode]
```
mode为intranode/internode/low_latency