require "nn"
local Rnn, parent = torch.class('nn.Rnn', 'nn.Module')

function Rnn:__init(module)
    if nil == module then
        error("Must supply a network to recur")
    end

    self.module = module
    -- API field 'output'. Do not change name.
    self.output = torch.Tensor()
    self.input_merged = torch.Tensor()
    self.grad_output_internal = torch.Tensor()
    self.gradInput = nil
end

---Compute the output by unrolling the network. Targets Y should be
--the width of the unrolling and not single points.
function Rnn:updateOutput(input)
    local module = self.module
    local xcols = module.dims.input[2]
    local unroll_len = input:size(2) - xcols + 1
    if unroll_len < 2 then
        error("Must unroll at least 2")
    end
    if unroll_len ~= ((input:size(2) - xcols) + 1) then
        error("Unroll length must match extension of input beyond base network")
    end

    -- For each unrolling of the network, shift the input to the left by one
    -- and copy the previous output into the rightmost column. Feed the shifted
    -- input into the next iteration of the unrolling.

    local input_merged = self.input_merged:resizeAs(input):narrow(
        2, 1, module.dims.input[2])
    input_merged:copy(input:narrow(2, 1, xcols))

    local wnd_prev = input_merged:narrow(2, 1, xcols - 1)
    local wnd_next = input_merged:narrow(2, 2, xcols - 1)
    local wnd_output = input_merged:narrow(2, xcols, 1)

    local output = self.output:resize(module.dims.input[1], unroll_len)
    local prev_output = output:narrow(2, 1, 1)
    prev_output:copy(module:forward(input_merged))

    for i = 2, unroll_len do
        -- Shift the input to the left.
        wnd_prev:copy(wnd_next)
        -- Copy the previous output into the rightmost column.
        wnd_output:copy(prev_output)

        -- Collect the current output into the merged output.
        local current_output = output:narrow(2, i, 1)
        current_output:copy(module:forward(input_merged))
        prev_output = current_output
    end 

    return output
end

function Rnn:updateGradInput(input, grad_output)
    local module = self.module
    local xcols = module.dims.input[2]
    local unroll_len = input:size(2) - xcols + 1
    if unroll_len < 2 then
        error("Must unroll at least 2")
    end
    if unroll_len ~= ((input:size(2) - xcols) + 1) then
        error("Unroll length must match extension of input beyond base network")
    end
    local output = self.output
    local grad_output_internal = self.grad_output_internal
        :resize(module.dims.input[1], unroll_len - 1)

    -- For each unrolling of the network going from output to input, add the
    -- gradient of the previous unrolling of the network and then backprop
    -- through the network. The true data intput is passed here, and so we
    -- must load the actual input computed by the unrolled network.

    -- Copy output to input in order to reconstruct the actual network input.
    local input_merged = self.input_merged:resizeAs(input):copy(input)
    input_merged
        :narrow(2, input:size(2) - unroll_len + 2, unroll_len - 1)
        :copy(output:narrow(2, 1, unroll_len - 1))

    local current_grad_output = grad_output:narrow(2, unroll_len, 1)
    local input_s_idx = input:size(2) - xcols + 1
    local input_actual = input_merged:narrow(2, input_s_idx, xcols)
    current_grad_output = module:updateGradInput(input_actual, current_grad_output)

    -- Walk backward over the merged input.
    for i = unroll_len - 1, 1, -1 do

        -- Take the gradInput for the last column and add it to the matching
        -- slice of gradOutput. This is because a split path causes an add on
        -- the gradient since influence from both paths joins together.
        current_grad_output = torch.add(grad_output_internal:narrow(2, i, 1),
                                        current_grad_output:narrow(2, xcols, 1),
                                        grad_output:narrow(2, i, 1))

        local input_actual = input_merged:narrow(2, input_s_idx - unroll_len + i, xcols)
        current_grad_output = module:updateGradInput(input_actual, current_grad_output)
    end

    self.gradInput = current_grad_output
    return current_grad_output
end

function Rnn:_acc_param_helper(action, input, grad_output, scale)
    local module = self.module
    local xcols = module.dims.input[2]
    local unroll_len = input:size(2) - xcols + 1
    if unroll_len < 2 then
        error("Must unroll at least 2")
    end
    if unroll_len ~= ((input:size(2) - xcols) + 1) then
        error("Unroll length must match extension of input beyond base network")
    end
    local output = self.output
    local grad_output_internal = self.grad_output_internal

    -- Copy output to input in order to reconstruct the actual network input.
    local input_merged = self.input_merged:resizeAs(input):copy(input)
    input_merged
        :narrow(2, input:size(2) - unroll_len + 2, unroll_len - 1)
        :copy(output:narrow(2, 1, unroll_len - 1))

    local current_grad_output = grad_output:narrow(2, unroll_len, 1)
    local input_s_idx = input:size(2) - xcols + 1
    local input_actual = input_merged:narrow(2, input_s_idx, xcols)
    current_grad_output = grad_output:narrow(2, unroll_len, 1)
    action(input_actual, current_grad_output, scale)

    -- Walk backward over the merged input.
    for i = unroll_len - 1, 1, -1 do
        local input_actual = input_merged:narrow(2, input_s_idx - unroll_len + i, xcols)
        current_grad_output = grad_output_internal:narrow(2, i, 1)
        action(input_actual, current_grad_output, scale)
    end
end

function Rnn:accGradParameters(input, grad_output, scale)
    scale = scale or 1

    self:_acc_param_helper(function(input, grad_output, lr)
        self.module:accGradParameters(input, grad_output, lr)
    end, input, grad_output, scale)
end

function Rnn:accUpdateGradParameters(input, grad_output, lr)
    self:_acc_param_helper(function(input, grad_output, lr)
        self.module:accUpdateGradParameters(input, grad_output, lr)
    end, input, grad_output, lr)
end

function Rnn:zeroGradParameters()
    self.module:zeroGradParameters()
end

function Rnn:updateParameters(learningRate)
    self.module:updateParameters(learningRate)
end

function Rnn:share(mlp,...)
    self.module:share(mlp,...)
end

function Rnn:reset(stdv)
    self.module:reset(stdv)
end

function Rnn:parameters()
    return self.module:parameters()
end

function Rnn:__tostring__()
    return self.module:__tostring__()
end

function Rnn._test()

    require "test_util"
    require "mid"
    require "models"
    require "Passthrough"

    local assert_equals = test_util.assert_equals
    local check_test = test_util.check_test

    local data_dir = "../midi"
    local stat, _ = pcall(function() lfs.dir(data_dir) end)
    if not stat then
        error("Data directory "..data_dir.." required")
    end

    -- Test Rnn:forward() with nn.Narrow.
    check_test(function()

        local unroll = 5
        local ds = mid.dataset.load_rnn(data_dir, "4/2-8-24-3-256", 10, unroll, 1)

        -- Repeat last column.
        do
            local narrow = models.append_ds_info(ds, nn.Narrow(2, 10, 1))
            local rnn = nn.Rnn(narrow)
            local input = ds.points[1][1]
            local output = rnn:forward(input)
            local diff = 0
            for i = 1, unroll do
                diff = diff + torch.abs(output:narrow(2, i, 1) - input:narrow(2, 10, 1)):sum()
            end
            assert_equals("Narrow should ouput end of input", 0, diff)
        end

        -- Repeat first column.
        do
            local narrow = models.append_ds_info(ds, nn.Narrow(2, 1, 1))
            local rnn = nn.Rnn(narrow)
            local input = ds.points[1][1]
            local output = rnn:forward(input)
            local diff = torch.abs(output - input:narrow(2, 1, unroll)):sum()
            assert_equals("Narrow should ouput beginning of input", 0, diff)
        end
    end)

    -- Test updateGradInput() by accumulating ones.
    check_test(function()

        local unroll = 7
        local ds = mid.dataset.load_rnn(data_dir, "4/2-8-24-3-256", 10, unroll, 1)

        local grad_output = torch.Tensor(ds.points[1][2]:size(1), unroll)
        local expect_sum = 0
        for i = 1, unroll do
            grad_output:narrow(2, i, 1):fill(i)
            expect_sum = expect_sum + i
        end

        local narrow = models.append_ds_info(ds, nn.Narrow(2, 10, 1))
        local rnn = nn.Rnn(narrow)
        local input = ds.points[1][1]
        rnn:updateOutput(input)
        local grad_output_actual = rnn:updateGradInput(input, grad_output)
        
        assert_equals(ds.points[1][1]:size(1), grad_output_actual:size(1))
        assert_equals(10, grad_output_actual:size(2))

        local grad_actual = grad_output_actual:narrow(2, 10, 1)
        local grad_expect = torch.Tensor():resizeAs(grad_actual):fill(expect_sum)
        assert_equals(0, torch.sum(grad_expect - grad_actual))

    end)
end

