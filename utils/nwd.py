def calculate_nwd(bbox1, bbox2, format_xywh=True): 
    # 定义计算Wasserstein距离的函数，输入两个边界框和格式标志
    bbox2 = bbox2.transpose()  
    # 转置第二个边界框，以匹配第一个边界框的维度
    
    if format_xywh:  # 如果边界框的格式是中心点坐标加宽高
        center_x1, center_y1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2  
        # 计算第一个边界框的中心点x，y坐标
        width1, height1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]  
        # 计算第一个边界框的宽度和高度
        center_x2, center_y2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2  
        # 计算第二个边界框的中心点x，y坐标
        width2, height2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]  
        # 计算第二个边界框的宽度和高度
    else:  # 如果边界框的格式是左上角坐标加宽高
        center_x1, center_y1, width1, height1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  
        # 直接使用第一个边界框的坐标和尺寸
        center_x2, center_y2, width2, height2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]  
        # 直接使用第二个边界框的坐标和尺寸
        
    dist_center_x = (center_x1 - center_x2) ** 2  
    # 计算中心点x坐标差的平方
    dist_center_y = (center_y1 - center_y2) ** 2  
    # 计算中心点y坐标差的平方
    distance_centers = dist_center_x + dist_center_y  
    # 计算中心点之间的距离（欧氏距离的平方）
    
    distance_width = ((width1 - width2) / 2) ** 2  
    # 计算宽度差的一半的平方
    distance_height = ((height1 - height2) / 2) ** 2  
    # 计算高度差的一半的平方
    distance_sizes = distance_width + distance_height  
    # 计算尺寸差异（宽度和高度差异的和）
    
    return distance_centers + distance_sizes  
    # 返回中心点差异和尺寸差异的总和作为Wasserstein距离