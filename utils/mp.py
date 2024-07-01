def do_multiprocessing(func, data, processes=20):       #do_multiprocessing 函数可以处理元组类型的数据，并确保每个子元组的长度相等
    import multiprocessing      # 导入多处理模块
    pool = multiprocessing.Pool(processes=processes)     # 创建一个具有指定进程数量的进程池

    # 检查数据类型是否为元组，如果是，确保每个元组的长度相同
    if isinstance(data, tuple):
        for item in data:
            assert len(item) == len(data[0])        # 确保每个元组的长度相同
        length = len(data[0]) // processes + 1      # 计算每个子任务的数据块大小
    else:
        length = len(data) // processes + 1         # 如果数据不是元组，计算每个子任务的数据块大小
    results = []
    for ids in range(processes):   # 循环创建子任务
        if isinstance(data, tuple):
            collect = (item[ids * length:(ids + 1) * length] for item in data)  # 分割数据
            results.append(pool.apply_async(func, (ids, *collect)))      # 异步应用函数到数据块
        else:
            collect = data[ids * length:(ids + 1) * length]
            results.append(pool.apply_async(func, (ids, collect)))
    pool.close()
    pool.join()     # 等待所有子进程完成
    collect = []
    for j, res in enumerate(results):   # 遍历所有结果
        ids, result = res.get()         # 获取结果
        assert j == ids                 # 确保结果的顺序与子任务的顺序一致
        collect.extend(result)          # 将结果合并到collect中
    return collect


def mp(func, data, processes=20, **kwargs):     #mp 函数更加通用，可以接受额外的关键字参数 **kwargs，并将这些参数传递给处理函数 func
    import multiprocessing   # 导入多处理模块
    pool = multiprocessing.Pool(processes=processes)
    length = len(data) // processes + 1         # 计算每个子任务的数据块大小
    results = []
    for ids in range(processes):
        collect = data[ids * length:(ids + 1) * length]     # 分割数据
        results.append(pool.apply_async(func, args=(collect, ), kwds=kwargs))  # 异步应用函数到数据块，传递附加参数
    pool.close()        # 异步应用函数到数据块，传递附加参数
    pool.join()         # 等待所有子进程完成
    collect = []
    for j, res in enumerate(results):   # 遍历所有结果
        result = res.get()
        collect.extend(result)
    return collect                      # 返回最终结果

#这两个函数的共同点是都使用多处理模块将大型数据集分割成多个子任务，并行处理，提高计算效率