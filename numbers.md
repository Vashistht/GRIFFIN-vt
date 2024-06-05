### 20 samples, t=.3, max_tokens = 64


FINAL RESULTS, for 20 requests, with select_only 2 and max_tokens 64:
rouge-1: 0.261620, rouge-2: 0.075396, rouge-l: 0.205441

FINAL RESULTS, for 20 requests, with select_only 4 and max_tokens 64:
rouge-1: 0.254364, rouge-2: 0.079299, rouge-l: 0.195337

FINAL RESULTS, for 20 requests, with select_only 6 and max_tokens 64:
rouge-1: 0.246636, rouge-2: 0.068545, rouge-l: 0.195369

FINAL RESULTS, for 20 requests, with select_only 8 and max_tokens 64:
rouge-1: 0.251608, rouge-2: 0.076325, rouge-l: 0.192071

FINAL RESULTS, for 20 requests, with select_only 12 and max_tokens 64:
rouge-1: 0.256835, rouge-2: 0.073455, rouge-l: 0.202085


FINAL RESULTS, for 20 requests, with select_only 16 and max_tokens 64:
rouge-1: 0.258728, rouge-2: 0.072984, rouge-l: 0.195409

FINAL RESULTS, for 20 requests, with select_only 20 and max_tokens 64:
rouge-1: 0.243615, rouge-2: 0.062146, rouge-l: 0.183115

FINAL RESULTS, for 20 requests, with select_only 24 and max_tokens 64:
rouge-1: 0.264634, rouge-2: 0.089232, rouge-l: 0.213648

FINAL RESULTS, for 20 requests, with select_only 32 and max_tokens 64:
rouge-1: 0.250811, rouge-2: 0.070185, rouge-l: 0.189729












### 20 samples, t=.3, max_tokens = 64
- use_cache =True, LLama base model
rouge-1: 0.254426, rouge-2: 0.081417, rouge-l: 0.202948

- forward, LLama base model
rouge-1: 0.254426, rouge-2: 0.081417, rouge-l: 0.202948

- use_cache =True, griffin .5
rouge-1: 0.250811, rouge-2: 0.070185, rouge-l: 0.189729

- forward, griffin .5
rouge-1: 0.250811, rouge-2: 0.070185, rouge-l: 0.189729

- use_cache =False, griffin .5
rouge-1: 0.254426, rouge-2: 0.081417, rouge-l: 0.202948
---
- use_cache =False, griffin .5, x[:,-1,:] 
rouge-1: 0.00, rouge-2: 0.00, rouge-l: 0.00 ## something wrong with slicing

    - checked the values make sense so idk something else under the hood 
    tensor([[[ 0.0190, -0.0076, -0.0084,  ..., -0.0287, -0.0079, -0.0186]]],
    tensor([[[ 0.0190, -0.0076, -0.0084,  ..., -0.0287, -0.0079, -0.0186]]],

- use_cache =False, griffin .5, %2==0
    rouge-1: 0.202492, rouge-2: 0.058870, rouge-l: 0.164388




---
### 5 samples, t=.3, max_tokens = 64
- can just use llama_og and get the same values with autoreg and cache=True
- get


---

### 5 samples, t=.3, max_tokens = 64
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

Works on griffin-sweep_test_rogue.py
    - rouge-1: 0.26586697119220853, rouge-2: 0.06810361828271567, rouge-l: 0.1891289093378677
    

### 100 samples, t=.3, max_tokens = 64

1. Forward=False, use_cache=True, 
    - **llama**: rouge-1: 0.241913, rouge-2: 0.072387, rouge-l: 0.199763
    - **llama_og**: rouge-1: 0.241913, rouge-2: 0.072387, rouge-l: 0.199763

2. Forward=False, use_cache=False, **llama_og** 
    - rouge-1: 0.271037, rouge-2: 0.092220, rouge-l: 0.227078

3. Forward=True, **llama**, autoreg_with_kv_cache
    - rouge-1: 0.241913, rouge-2: 0.072387, rouge-l: 0.199763

4. Llama (Llama original)
- rouge-1: 0.270757, rouge-2: 0.091888, rouge-l: 0.226346
---

density = .1, use_cache=False

    - llama_og: rouge-1: 0.118528, rouge-2: 0.000000, rouge-l: 0.090810
    - 



    tensor([[[-0.0040,  0.0070, -0.0011,  ..., -0.0007, -0.0003, -0.0002]]],