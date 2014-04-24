local SgdMomentum = torch.class('nn.SgdMomentum')

function SgdMomentum:__init(module, criterion)
   self.learningRate = 1e-2
   self.learningRateDecay = 0.3
   self.maxIteration = 100
   self.convergeEps = 1e-6
   self.momentum = 0.9
   self.shuffleIndices = true
   self.module = module
   self.criterion = criterion
end

function SgdMomentum:train(dataset)

   local currentLearningRate = self.learningRate
   local module = self.module
   local criterion = self.criterion

   local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
   if not self.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
   end

   local errPrev = math.huge

   -- Initialize previous weights used to compute momentum.
   criterion:forward(module:forward(dataset[shuffledIndices[1]][1]),
                                    dataset[shuffledIndices[1]][2])
   local parameters, gradients = module:parameters()
   local prevParams = {}
   for _, w in ipairs(parameters) do
      table.insert(prevParams, w:clone())
   end

   print("# SgdMomentum: training")

   local iteration = 1
   while true do

      local currentError = 0
      for t = 1,dataset:size() do
         local example = dataset[shuffledIndices[t]]
         local input = example[1]
         local target = example[2]

         currentError = currentError + criterion:forward(module:forward(input), target)
         module:updateGradInput(input, criterion:updateGradInput(module.output, target))

         -- Add momentum term to gradients.
         local parameters, gradients = module:parameters()
         for i, gw in ipairs(gradients) do
            local w = parameters[i]
            local wPrev = prevParams[i]
            -- -\delta_w = w_{t-1} - w_t
            wPrev:add(-1, w)
            -- Update gradient with momentum term correcting for sign and learning rate.
            gw:add(-self.momentum / currentLearningRate, wPrev)
            wPrev:copy(w)
         end

         module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

         if self.hookExample then
            self.hookExample(self, example)
         end
      end

      if self.hookIteration then
         self.hookIteration(self, iteration)
      end

      currentError = currentError / dataset:size()
      print("# current error = "..currentError)
      currentLearningRate = self.learningRate / (1 + (iteration * self.learningRateDecay))

      if self.maxIteration > 0 and iteration >= self.maxIteration then
         print("# SgdMomentum: you have reached the maximum number of iterations")
         break
      end

      -- Check convergence (expect decrease).
      local errDelta = errPrev - currentError
      if errDelta < self.convergeEps then
         print("# SgdMomentum: converged after "..iteration.." iterations")
         break
      end
      errPrev = currentError

      iteration = iteration + 1
   end
end
