# Pruning from scratch #

## Aim ##
- 通过BN层的 \\( \gamma \\)，把 \\( \gamma \\)的当作成门函数来实现剪枝的目的
- 实现在使用随机初始化权重的时候我们的方法依然能够得到比较好的剪枝结果。

## prepare ##
- pytorch \\( \geq \\) 1.0.1
- VGG19
- CIFAR10(60000 ,32*32, 10class, 50000 for train,1000 for test)

## How to achive ##
### Model生成 ###
- 所有初始化的BN层的权重设成0.5，以剪去50%的FLOPS
- 随机初始化的权重（torch.normal，随机生成函数）
```python
def _initialize_weights(self):
	for m in self.modules():
	    if isinstance(m, nn.Conv2d):
	        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(0.5)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
```
### 架构的训练 ###
- 首先将网络的权重冻结，只留下BN层的权重，代码如下：
```python
 for para in model.parameters():
        para.requires_grad = False
 for layer in model.modules():
    if isinstance(layer, nn.BatchNorm2d):
        for para in layer.parameters():
            para.requires_grad = True
```

- 再用10个epoch训练出可以网络架构，代码如下

```python
    for epochid in range(args.aepoch):
        print('==> Epoch: %d' % epochid)
        train_loss = 0.0
        total = 0
        correct = 0
        for batchid, (data, target) in enumerate(train_loader):
            if args.Use_Cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            regularzation_update(model, args)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            avg_loss = train_loss / (batchid+1)
            acc = correct / total
            progress_bar(batchid, len(train_loader), 'Loss: %.3f | Acc: %.3f'% (avg_loss, acc))
```

- 注意中间`regularzation_update(model, args)`，它的作用是用一个正则化项去更新梯度，正则化项为
$$ \Omega (\Lambda)=(\frac{\sum_j|\Lambda|_1}{\sum_jC_j}-r)^2 $$
代码如下：

```python
def regularzation_update(model, args):
    if not args.sum_channel:
        args.sum_channel = 0
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                args.sum_channel += layer.weight.size()[0]
    sumc = args.sum_channel
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.grad.data.add(args.balance * 2.0 * torch.sign(layer.weight.data)*(layer.weight.data/sumc-args.ratio))
```
### 剪枝操作 ###
- 剪枝所用的模型就是上述用10个epoch训练出来的模型
- 剪枝操作去掉的只是BN层的权重，BN层每个权重对应的是一个channel，即每一个权重相当于这个channel的gates。将BN层权重去掉本质上就是把这个channel去掉。
- 剪枝过程中是将不同层的BN层放在一起比较，所以是一个全局剪枝的过程（global pruning）
-确定gates的代码如下

```python
for lid, layer in enumerate(model.modules()):
        if isinstance(layer, nn.BatchNorm2d):
            nchannel = layer.weight.data.shape[0]
            gates[index:index+nchannel] = layer.weight.data.abs().clone()
            index += nchannel
```

- 然后定义一个阈值threshold ，使得小于这个阈值的gates全部为0，大于这个阈值的保留。但是这个阈值的设定不能是机械式设定。本作中应用了二分法，动态的搜寻最符合要求的阈值。代码如下：

```python
def binary_search(model, gates, args, data_loader):
    # Get single batch data to profile the flops of the model 
    model = copy.deepcopy(model).cpu()
    data, target = next(iter(data_loader))
    ori_macs, ori_params = profile(model, inputs=(data,))  #确定模型原始的的FLOPs和Params
    #pos = min(int(len(gates) * args.ratio), len(gates)-1)
    sorted_gates, _ = torch.sort(gates)       #将所有们由大到小排列
    # TODO: use binary search to find the threshold for the pruning
    lpos, rpos = 0, len(sorted_gates) - 1     #定义一个最大值和一个最小值
    input = torch.randn((args.batchsize, 3))  #YYSY这一行真的没用
    eps = 1
    macs, params = None, None
    while lpos < rpos:
        midpos = int((lpos + rpos) / 2)
        cur_thres = sorted_gates[midpos]      #求暂时的阈值
        cfg = [] 
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                weight_copy = layer.weight.data.abs().clone()
                mask = weight_copy.gt(cur_thres)    #根据阈值生成mask，gt是大于   
                cfg.append(int(torch.sum(mask).item()))
            elif isinstance(layer, nn.MaxPool2d):
                cfg.append('M')
        pruned_model = models.__dict__[args.model](args.num_class, cfg=cfg) #根据阈值生成的暂时的模型，目的是为了计算FLOPs和Params
        pruned_model(data)
        macs, params = profile(pruned_model, inputs=(data,)) #得到两个新参数
        if abs(macs - ori_macs * args.ratio) < eps:
            lpos = midpos
            break
        elif macs > ori_macs * args.ratio:
            lpos = midpos + 1
        else: 
            #macs < ori_macs * args.ratio:
            rpos = midpos - 1
    print('==>Original Model:')
    print('  Flops: {}G  Parameters: {}M'.format(ori_macs/(10**9), ori_params/(10**6)))
    print('==>Pruned Model:')
    print('  Flops: {}G  Parameters: {}M'.format(macs/(10**9), params/(10**6)))
    return sorted_gates[lpos]
```

-根据阈值生成mask，把mask与BN层权重相乘即可得到结果，代码如下：

```python
for lid, layer in enumerate(model.modules()):
        if isinstance(layer, nn.BatchNorm2d):
            weight_copy = layer.weight.data.abs().clone()
            threshold=threshold.cuda()
            mask = weight_copy.gt(threshold)
            mask = mask.float().cuda()
            layer.weight.data.mul_(mask)
            layer.bias.data.mul_(mask)
            pruned += mask.shape[0] - sum(mask)
            cfg.append(int(torch.sum(mask).item()))
            cfg_mask.append(mask)
        elif isinstance(layer, nn.MaxPool2d):
            cfg.append('M')
```
### 新模型的权重训练 ###
-再把train_loader，val_loader放进去训练模型。训练步骤为在每一个epoch里面model.train()一次，得到loss并loss.backward(),再用验证集去model.eval()。注意model.train()和model.eval()的区别。model.train()启用 BatchNormalization 和 Dropout，model.eval()不启用。代码如下:

```python
def weight_train(model, train_loader, val_loader, args):
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(args.wepoch*0.5), int(args.wepoch*0.75)], gamma=0.1)  #MultiStepLR:调整学习率
    for i in range(args.wepoch):
        print('==>Epoch %d' % (i+1))
        print('==>Training')
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batchid, (data, target) in enumerate(train_loader):
            if args.Use_Cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += output.size(0)
            correct += predicted.eq(target).sum().item()
            avg_loss = train_loss / (batchid + 1)
            avg_acc = correct / total
            progress_bar(batchid, len(train_loader), 'Loss: %.3f | Acc: %.3f' % (avg_loss, avg_acc))
        # Validation
        print('==>Validating')
        val_acc = validation(model, val_loader, criterion, args.Use_Cuda)    
        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint = {'state_dict':model.state_dict(), 'Acc':best_acc}
            fname = os.path.join(args.outdir, 'best.pth.tar')
            torch.save(best_checkpoint, fname)
        print('==>Best validation accuracy', best_acc)
        # Save checkpoint
        if (i + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outdir, 'checkpoint.pth.tar')) 
        # Lr_scheduler
        if args.lr_decay:
            lr_scheduler.step()
```


