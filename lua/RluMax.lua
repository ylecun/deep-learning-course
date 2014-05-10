require "nn"
local RluMax = torch.class('nn.RluMax', 'nn.Module')

function RluMax:__init()
   self.mask = torch.Tensor()
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

-- Allow only positive values.
function RluMax:updateOutput(input)
   local mask = self.mask:resizeAs(input)
   local output = self.output:resizeAs(input)

   output:cmul(input, torch.gt(mask, input, 0))

   return output
end

-- Gradient is constant for positive values and zero otherwise.
function RluMax:updateGradInput(input, gradOutput)
   local mask = self.mask:resizeAs(input)
   local gradInput = self.gradInput:resizeAs(input)

   -- Use subgradient 1 at value 0.
   gradInput:cmul(torch.ge(mask, input, 0), gradOutput)

   return gradInput
end
