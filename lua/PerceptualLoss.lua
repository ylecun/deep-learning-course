require "nn"
require "math"

local PerceptualLoss, parent = torch.class('PerceptualLoss', 'nn.Criterion')

---Takes to_byte in order to suppress any gradient when
--the resulting conversion to byte has zero loss.
function PerceptualLoss:__init(to_byte, lambda, alpha)
    parent.__init(self)

    self.sizeAverage = true
    self.lambda = lambda or 2
    self.alpha = alpha or 2
    self.t1 = torch.Tensor()
    self.mask = torch.Tensor()

    if nil == to_byte then
       error("Must supply to_byte() function")
    end
    self.to_byte = to_byte

    self.mask_func = function(_, xx, yy)
       local X = to_byte(xx)
       local Y = to_byte(yy)
       local diff = X - Y
       return diff ~= 0 and 1 or 0
    end

end

--- We wish to penalize much more highly missed or extraneous notes.
--    l = abs(x - y) + (\lambda * abs(x - y))^\alpha
function PerceptualLoss:updateOutput(input, target)

    local X = input
    local Y = target
    local t1 = self.t1:resizeAs(input)
    local mask = self.mask:resizeAs(input)

    mask:map2(X, Y, self.mask_func)

    t1:add(X, -1, Y):abs():cmul(mask)
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
    local mask = self.mask:resizeAs(input)
    local gradInput = self.gradInput:resizeAs(input)

    mask:map2(X, Y, self.mask_func)

    -- Gradient of L1 is the sign of the difference.
    t1:add(X, -1, Y):cmul(mask)
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
