require "nn"
require "math"

local PerceptualLoss, parent = torch.class('PerceptualLoss', 'nn.Criterion')

function PerceptualLoss:__init(lambda, alpha)
    parent.__init(self)
    self.sizeAverage = true
    self.lambda = lambda or 2
    self.alpha = alpha or 2
    self.t1 = torch.Tensor()
    self.t2 = torch.Tensor()
end

--- We wish to penalize much more highly missed or extraneous notes.
--    l = abs(x - y) + (\lambda * abs(x - y))^\alpha
function PerceptualLoss:updateOutput(input, target)

    local X = input
    local Y = target
    local t1 = self.t1:resizeAs(input)
    local t2 = self.t2:resizeAs(input)

    t1:add(X, -1, Y):abs()
    local loss_abs = t1:sum()
    local loss_pow = t1:pow(self.alpha):sum() * self.lambda^(self.alpha)
    local loss = loss_abs + loss_pow

    if self.sizeAverage then
        loss = loss / input:nElement()
    end

    --print("loss", loss, loss_abs, loss_pow, X:max(), Y:max())

    return loss
end

function PerceptualLoss:updateGradInput(input, target)

    local X = input
    local Y = target
    local t1 = self.t1:resizeAs(input)
    local t2 = self.t2:resizeAs(input)
    local gradInput = self.gradInput:resizeAs(input)

    -- Gradient of L1 is the sign of the difference.
    t1:add(X, -1, Y)
    torch.sign(gradInput, t1)

    -- Here is input to polynomial.
    local scale = self.alpha * self.lambda * self.lambda^(self.alpha - 1)
    t1:abs():pow(self.alpha - 1):mul(scale)

    -- Scale by derivative of polynomial expression and add to grad.
    gradInput:addcmul(gradInput, t1)

    if self.sizeAverage then
        gradInput:mul(1.0 / input:nElement())
    end

    --print("grad", gradInput:max(), gradInput:min(), X:max(), Y:max())

    return gradInput
end
