from torchmetrics import R2Score

def compute_r2(y_true, y_predicted):
    # np_sse = sum((y_true - y_predicted)**2)
    # np_tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    # np_r2_score = 1 - (np_sse / np_tse)
    # print(f"type(y_true) = {type(y_true)} , type(y_predicted) = {type(y_predicted)}")
    # print(f"sse = {sse}, tse = {tse}, r2_score = {r2_score}")
    # print(f"type(r2_score) = {type(r2_score)}, type(sse) = {type(sse)}, type(tse) = {type(tse)}")
    # assert False
    # return r2_score, sse, tse

    
    y_true = torch.from_numpy(y_true)
    y_predicted = torch.from_numpy(y_predicted)
    
    target_mean = torch.mean(y_true)
    sse = torch.sum((y_true - y_predicted) ** 2)
    tse = torch.sum((y_true - target_mean) ** 2)
    r2 = 1 - sse / tse

    # r2score = R2Score()
    # print(f"np_r2_score = {np_r2_score}, np_tse = {np_tse}, np_sse = {np_sse}")
    # print(f"r2 = {r2}, tse = {tse}, sse = {sse}")
    # print(f"r2score() = {r2score(y_predicted, y_true)}")
        
    # assert r2score(y_predicted, y_true) == np_r2_score,f"r2:{np_r2_score}, r2score(y_predicted, y_true):{r2score(y_predicted, y_true)}"
    
    # assert False
    
    return  r2, sse, tse
    

# def r2_loss(y_predicted, y_true):
#     """
#     From https://en.wikipedia.org/wiki/Coefficient_of_determination
#     """
#     target_mean = torch.mean(y_true)
#     sse = torch.sum((y_true - y_predicted) ** 2)
#     tse = torch.sum((y_true - target_mean) ** 2)
#     r2 = 1 - sse / tse
#     return r2