require "nn"
require "math"

require "test_util"

local PerceptualLoss, parent = torch.class('PerceptualLoss', 'nn.Criterion')

--- Initialize with function taking NN output value to a note.
function PerceptualLoss:__init(to_byte, lambda, alpha, epsilon)
    parent.__init(self)
    self.sizeAverage = true
    self.lambda = lambda or 5
    self.alpha = alpha or 4
    --self.epsilon = epsilon or 0.5
    self.X = torch.Tensor()
    self.Y = torch.Tensor()
    self.t1 = torch.Tensor()
    self.t2 = torch.Tensor()

    self.map_to_byte = function(a, b)
        return to_byte(b)
    end
end

--- We wish to penalize much more highly missed or extraneous notes.
--
--   l = abs(x - y) + abs(x - y)^\alpha
--
--   l = abs(x - y) + exp(abs(x - y)^\alpha) - 1
function PerceptualLoss:updateOutput(input, target)

    local X = input --self.X:resizeAs(input)
    local Y = target --self.Y:resizeAs(input)
    local t1 = self.t1:resizeAs(input)
    local t2 = self.t2:resizeAs(input)

    --X:map(input, self.map_to_byte)
    --Y:map(target, self.map_to_byte)

    t1:add(X, -1, Y):abs()
    local loss_abs = t1:sum()
    local loss_pow = t1:pow(self.alpha):sum() * self.lambda^(self.alpha)
    local loss = loss_abs + loss_pow

--    t1:add(X, -1, Y):abs()
--    local loss_abs = t1:sum()
--    local loss_exp = t1:pow(self.alpha):exp():sum()
--    local loss = loss_abs + loss_exp - input:nElement()

    if self.sizeAverage then
        loss = loss / input:nElement()
    end

    return loss
end

function PerceptualLoss:updateGradInput(input, target)

    --print("input:size(): "..tostring(input:size()))
    --print("target:size(): "..tostring(target:size()))
    local X = input --self.X:resizeAs(input)
    local Y = target --self.Y:resizeAs(input)
    local t1 = self.t1:resizeAs(input)
    local t2 = self.t2:resizeAs(input)
    local gradInput = self.gradInput:resizeAs(input)

    --X:map(input, self.map_to_byte)
    --Y:map(target, self.map_to_byte)

    -- Gradient of L1 is the sign of the difference.
    t1:add(X, -1, Y)
    torch.sign(gradInput, t1)

    -- Here is input to polynomial.
    local scale = self.alpha * self.lambda * self.lambda^(self.alpha - 1)
    t1:abs():pow(self.alpha - 1):mul(scale)

    -- Scale by derivative of polynomial expression and add to grad.
    gradInput:addcmul(gradInput, t1)

--    -- Gradient of L1 is the sign of the difference.
--    t1:add(X, -1, Y)
--    torch.sign(gradInput, t1)
--
--    -- Gradient of argument to exp().
--    t1:abs():pow(self.alpha - 1):mul(self.alpha):cmul(gradInput)
--    -- Value of exp().
--    t2:add(X, -1, Y):abs():pow(self.alpha):exp()
--    -- Gradient of entire exp() expression.
--    t1:cmul(t2)
--
--    -- Sum both gradients.
--    gradInput:add(t1)

    if self.sizeAverage then
        gradInput:mul(1.0 / input:nElement())
    end

    return gradInput
end
