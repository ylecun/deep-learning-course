require "nn"
local Passthrough = torch.class('nn.Passthrough', 'nn.Module')

function Passthrough:__init()
end

function Passthrough:updateOutput(input)
   return input
end

function Passthrough:updateGradInput(input, gradOutput)
   return gradOutput
end
