
#TODO
import torch
from models.base_model import DomainDisentangleModel
from my_loss import DomainDisentangleLoss


class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Loss functions
        # self.criterion = DomainDisentangleLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        # Setup optimization procedure
        # self.optimizer = torch.optim.Adam(list(self.model.parameters())+list(self.criterion.parameters()), lr=opt['lr'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        # self.optimizer2 = torch.optim.Adam(self.criterion.parameters(), lr=opt['lr'])
        print("model parameters: ",self.model.parameters())
        print("criterion parameters: ",self.criterion.parameters())
    def save_checkpoint(self, path, epoch, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['epoch'] = epoch
        checkpoint['iteration'] = iteration  # 当前第几个iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return epoch, iteration, best_accuracy, total_train_loss
    def train_iteration(self, data):
        # [xd]: source/target的图
        # [y]: source的category label(狗，猫...)，如果当前x是target的图则y为-1
        # [yd]: source+target的domain label (cartoon, photo...)
        xd, y, yd = data # xd包含了 source domain和target domain的图，如果是target domain，则对应的y是-1
        # x = xd[y!=-1,:,:,:].detach()

        # x = x.to(self.device)
        y = y.to(self.device)
        xd = xd.to(self.device)
        yd = yd.to(self.device)

        fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds = self.model(xd)

        Cfcs = Cfcs[y != -1, :]
        DCfcs = DCfcs[y != -1, :]  # 删除掉没有label的数据的处理结果，使其不参加loss计算
        # print(y)
        y = y[y != -1]
        y = y.type(torch.int64)
        if y.shape[0] == 0:
            return 0
        # loss = self.criterion(fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y, yd)  # 这里要重新写！！！ 不能 直接模型处理完结果传到损失函数里，损失函数可能用的总体模型中不同阶段的输出，而不是最终整体的输出！！！！
        # print(Cfcs.requires_grad)
        loss = self.criterion(Cfcs,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, loader):
        self.model.eval()  # 设置为evaluation 模式
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():  # 禁用梯度计算，即使torch.tensor(xxx,requires_grad = True) 使用.requires_grad()也会返回False
            for x, y, yd in loader:  # type(x) tensor

                y = y.to(self.device)
                x = x.to(self.device)
                yd = yd.to(self.device)

                fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds = self.model(x)
                # Cfcs = Cfcs[y != -1, :].detach()
                # DCfcs = DCfcs[y != -1, :].detach()  # 删除掉没有label的数据的处理结果，使其不参加loss计算
                # y = y[y != -1].detach()
                # y = y.type(torch.int64)
                # if y.shape[0] != 0:
                # loss += self.criterion(fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y,yd)  # 这里要重新写！！！ 不能 直接模型处理完结果传到损失函数里，损失函数可能用的总体模型中不同阶段的输出，而不是最终整体的输出！！！！
                loss += self.criterion(Cfcs,y)
                pred = torch.argmax(Cfcs, dim=-1)
                # print(Cfcs.shape)
                accuracy += (pred == y).sum().item()
                count += x.size(0)
                # print(count)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss