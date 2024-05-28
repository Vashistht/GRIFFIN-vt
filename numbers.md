# 5 samples, t=.3, max_tokens = 64
1. Forward=False, use_cache=True, **llama_og**
    - rouge-1: 0.265867, rouge-2: 0.068104, rouge-l: 0.189129

2. Forward=False, use_cache=False, **llama_og** 
    - essentially doing expert selection every time
    - rouge-1: 0.255716, rouge-2: 0.069691, rouge-l: 0.181760
---
3. Forward=False, use_cache=True,**llama**
    - rouge-1: 0.265867, rouge-2: 0.068104, rouge-l: 0.189129

4. Forward=False, use_cache=False,**llama**
    - rouge-1: 0.201757, rouge-2: 0.019048, rouge-l: 0.116854

5. Forward=True,**llama**
    - rouge-1: 0.201757, rouge-2: 0.019048, rouge-l: 0.116854

6. Forward=True, **llama**, autoreg_with_kv_cache
    - rouge-1: 0.265867, rouge-2: 0.068104, rouge-l: 0.189129

---
Conclusion:
    - works with kv cache!
    - 

Works on griffin-sweep_test_rogue.py
    - rouge-1: 0.26586697119220853, rouge-2: 0.06810361828271567, rouge-l: 0.1891289093378677
    