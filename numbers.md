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