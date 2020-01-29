import torch
import torch.distributions


class RandomSignAttack:
    def __init__(self, epsilon=0.3):
        self.noise = torch.distributions.normal.Normal(loc=0, scale=1)
        self.epsilon = epsilon

    def generate(self, x):
        batch_noise = self.noise.sample(x.shape).sign().to(x.device)
        adv_x = torch.clamp(x.detach() + self.epsilon * batch_noise, 0, 1)

        return adv_x


class FGSMAttack:
    def __init__(self, model, criterion, epsilon=0.3):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def generate_v0(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        loss.backward()

        adv_x = torch.clamp(x.detach() + self.epsilon * torch.sign(x.grad.detach()), 0, 1).detach()

        return adv_x

    def generate(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        x_grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

        adv_x = torch.clamp(x.detach() + self.epsilon * torch.sign(x_grad.detach()), 0, 1).detach()

        return adv_x


class LLFGSMAttack:
    def __init__(self, model, criterion, epsilon=0.3):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def generate(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.min(1)[1] if y is None else y)

        x_grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

        adv_x = torch.clamp(x.detach() - self.epsilon * torch.sign(x_grad.detach()), 0, 1).detach()

        return adv_x


class RandomFGSMAttack:
    def __init__(self, model, criterion, alpha=0.15, epsilon=0.3):
        self.model = model
        self.criterion = criterion
        self.alpha = alpha
        self.epsilon = epsilon
        self.noise = torch.distributions.normal.Normal(loc=0, scale=1)

    def generate(self, x, y=None):
        batch_noise = self.noise.sample(x.shape).sign().to(x.device)
        x_p = torch.clamp(x.detach() + self.alpha * batch_noise, 0, 1).detach()

        x_p.requires_grad = True
        output = self.model(x_p)
        loss = self.criterion(output, self.model(x).max(1)[1] if y is None else y)

        x_p_grad = torch.autograd.grad(loss, x_p, only_inputs=True)[0]

        adv_x = torch.clamp(x_p.detach() + (self.epsilon-self.alpha) * torch.sign(x_p_grad.detach()), 0, 1).detach()

        return adv_x


class IterativeFGSMAttack:
    def __init__(self, model, criterion, epsilon=0.3, num_steps=40, step_size=0.01):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def generate(self, x, y=None):
        adv_x = x.clone().detach()

        if y is None:
            y = self.model(x.detach()).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x.detach() + self.step_size * torch.sign(grad.detach())

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class PGDAttack:
    def __init__(self, model, criterion, epsilon=0.3, num_steps=40, step_size=0.01):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def generate(self, x, y=None):
        adv_x = x.clone().detach()
        adv_x += torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1)

        if y is None:
            y = self.model(x.detach()).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x.detach() + self.step_size * torch.sign(grad.detach())

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x