
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
        self.nll_loss = torch.nn.NLLLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.rec_loss = torch.nn.MSELoss()
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.1,device='cuda'), requires_grad=True)
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.1,device='cuda'), requires_grad=True)
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(list(self.model.parameters())+[self.alpha1,self.alpha2], lr=opt['lr'])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        # self.optimizer2 = torch.optim.Adam(self.criterion.parameters(), lr=opt['lr'])
        # print("model parameters: ",self.model.parameters())
        # print("criterion parameters: ",self.criterion.parameters())
    def entropy_loss(self, f):  # 应该返回一个标量 最后是求和的

        # f = torch.clamp_min_(f,0.0001)
        # print(f,)
        logf = torch.log(f)
        mlogf = logf.mean(dim=0)
        return -mlogf.sum()
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
    def train_iteration(self, data_source, data_target):
        # [x]: source/target的图
        # [y]:  category label(狗，猫...)
        # [yd]: domain label (cartoon, photo...)

        x_s, y_s, yd_s = data_source
        # print(y_s.size())
        x_t, _, yd_t = data_target

        x_s = x_s.to(self.device)
        y_s = y_s.to(self.device)
        yd_s = yd_s.to(self.device)

        x_t = x_t.to(self.device) # [32,3,224,224] 32是一个batch中图片数量
        yd_t = yd_t.to(self.device)

        # x = torch.cat((x_s,x_t),0) # [64,3,224,224]
        # yd = torch.cat((yd_s,yd_t),0) # [64] 前32个是source domain label， 后32个是target

        fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds = self.model(x_s)
        # loss = self.criterion(fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y_s, yd)  # 这里要重新写！！！ 不能 直接模型处理完结果传到损失函数里，损失函数可能用的总体模型中不同阶段的输出，而不是最终整体的输出！！！！
        l_class = self.cross_entropy(Cfcs,y_s)
        # DCfcs = torch.clamp_min_(DCfcs,0.0001)
        l_class_ent_1 = self.entropy_loss(DCfcs)

        l_domain_1 = self.cross_entropy(DCfds, yd_s)
        # Cfds = torch.clamp_min_(Cfds,0.0001)
        l_domain_ent_1 = self.entropy_loss(Cfds)

        l_rec_1 = self.rec_loss(fG,fG_hat)


        fG, fG_hat, _, _, DCfds, Cfds = self.model(x_t)

        # DCfcs = torch.clamp_min_(DCfcs,0.0001)
        # l_class_ent_2 = self.entropy_loss(DCfcs) 注释掉这个
        L_class = l_class + self.alpha1* (l_class_ent_1 )

        l_domain_2 = self.cross_entropy(DCfds,yd_t)
        l_domain = l_domain_1 + l_domain_2

        # Cfds = torch.clamp_min_(Cfds,0.0001)
        l_domain_ent_2 = self.entropy_loss(Cfds)
        L_domain = l_domain + self.alpha2*(l_domain_ent_1 + l_domain_ent_2)

        l_rec_2 = self.rec_loss(fG,fG_hat)
        L_rec = l_rec_1 + l_rec_2

        loss =self.w1 * L_class + self.w2 * L_domain + self.w3 * L_rec
        # if loss < 0 :
        print(L_class,'-',l_class_ent_1)
        print(L_domain,'-',l_domain_ent_1,'-',l_domain_ent_2)
            # print(L_rec)
        # print(loss,L_domain)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, loader):
        self.model.eval()  # 设置为evaluation 模式
        print(self.alpha1,self.alpha2)
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():  # 禁用梯度计算，即使torch.tensor(xxx,requires_grad = True) 使用.requires_grad()也会返回False
            for x, y, yd in loader:  # type(x) tensor

                y = y.to(self.device)
                x = x.to(self.device)
                yd = yd.to(self.device)

                fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds = self.model(x)
                # opt = parse_arguments()
                # Cfcs = Cfcs[0:opt['batch_size'], :]
                # loss += self.criterion(fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y,yd)  # 这里要重新写！！！ 不能 直接模型处理完结果传到损失函数里，损失函数可能用的总体模型中不同阶段的输出，而不是最终整体的输出！！！！

                loss += self.cross_entropy(Cfcs, y)
                # L_domain = self.nll_loss(torch.log(DCfds), yd)  # + self.alpha2*self.entropy_loss(Cfds)
                # L_rec = self.rec_loss(fG, fG_hat)
                # loss += self.w1 * L_class + self.w2 * L_domain + self.w3 * L_rec

                pred = torch.argmax(Cfcs, dim=-1)
                # print(Cfcs.shape)
                accuracy += (pred == y).sum().item()
                count += x.size(0)
                # print(count)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss