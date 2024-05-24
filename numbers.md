# Griffin with my llama.py
## First 50 samples
FINAL RESULTS
rouge-1: 0.094043, rouge-2: 0.020702, rouge-l: 0.074414
## First 100 samples
FINAL RESULTS
rouge-1: 0.108260, rouge-2: 0.021357, rouge-l: 0.085131
- seems like the llama.py is bad


## forward (sweep.py)
- (10, max_tokens 64 with set_epoch==0 and first line)
rouge-1: 0.1417299321473282, rouge-2: 0.015855490937714073, rouge-l: 0.11258578800318406
- Inside:
rouge-1: 0.1947776265314904 rouge-2: 0.05431682768070919 rouge-l: 0.14421382789965592


## means my expert selection is not right (50)
FINAL RESULTS
rouge-1: 0.239113, rouge-2: 0.079522, rouge-l: 0.196250
- (10, max_tokens 128)
rouge-1: 0.188634, rouge-2: 0.037342, rouge-l: 0.131548
- (10, max_tokens 64)
rouge-1: 0.224498, rouge-2: 0.049714, rouge-l: 0.161488
rouge-1: 0.224498, rouge-2: 0.049714, rouge-l: 0.161488

- (10, max_tokens 64, with set_epoch==0)
rouge-1: 0.224498, rouge-2: 0.049714, rouge-l: 0.161488

