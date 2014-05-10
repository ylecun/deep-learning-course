require "nn"
require "math"

local PerceptualLoss, parent = torch.class('nn.PerceptualLoss', 'nn.Criterion')

---Takes to_byte in order to suppress any gradient when
--the resulting conversion to byte has zero loss.
function PerceptualLoss:__init(to_byte, perceptualThresh)
    parent.__init(self)

    self.alpha = alpha or 2
    self.t1 = torch.Tensor()
    self.mask = torch.Tensor()
    self.perceptualThresh = perceptualThresh or 4 / 255

    if nil == to_byte then
        error("Must supply to_byte() function")
    end

    self.loss_func = function(_, xx, yy)
        local X = to_byte(xx)
        local Y = to_byte(yy)
        local diff = math.abs(xx - yy)

        -- Polynomial loss when X is 0 and Y > 0.
        if X == 0 and Y > 0 then
            return diff * 2
        elseif diff > self.perceptualThresh then
            return diff
        else
            return 0
        end
    end

    self.grad_func = function(_, xx, yy)
        local X = to_byte(xx)
        local Y = to_byte(yy)
        local diff = math.abs(xx - yy)
        local sign = xx >= yy and 1 or -1

        -- Polynomial loss when X is 0 and Y > 0.
        if X == 0 and Y > 0 then
            return sign * 2
        elseif diff > self.perceptualThresh then
            return sign
        else
            return 0
        end
    end
 end

function PerceptualLoss:updateOutput(input, target)

    local X = input
    local Y = target
    local t1 = self.t1:resizeAs(input)

    local loss = t1:map2(X, Y, self.loss_func):sum()

    --local xxmax = X:max()
    --if xxmax > -1 then
    --    print(xxmax, loss)
    --end

    return loss
end

function PerceptualLoss:updateGradInput(input, target)

    local X = input
    local Y = target
    local gradInput = self.gradInput:resizeAs(input)

    gradInput:map2(X, Y, self.grad_func)

    return gradInput
end
