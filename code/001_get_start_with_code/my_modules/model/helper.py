import torch

do_gpu = True
def move_to_gpu(item):
    
    if(torch.cuda.is_available() and do_gpu ):
#         print('moving to GPU',torch.cuda.get_device_name(0))
        return item.cuda()
    else:
        return item