require "os"
require "math"
require "string"
require "nn"
local SgdMomentum = torch.class('nn.SgdMomentum')

function SgdMomentum:__init(module, criterion, kwargs)

    self.module = module
    self.criterion = criterion

    kwargs = kwargs or {}
    self.learning_rate = kwargs.learning_rate or 1e-5
    self.learning_rate_decay = kwargs.learning_rate_decay or 0.1
    self.max_iteration = kwargs.max_iteration or 5
    self.converge_eps = kwargs.converge_eps or 1e-6
    self.momentum = kwargs.momentum or 0.8
    self.shuffle_indices = kwargs.shuffle_indices or true
    self.mini_batch_size = kwargs.mini_batch_size or 5000
end

---Convert time in seconds h
local s_to_ddmmhhss = function(time_s)

    local day = math.floor(time_s / (24 * 60 * 60))
    time_s = time_s - (day * 24 * 60 * 60)
    local hour = math.floor(time_s / (60 * 60))
    time_s = time_s - (hour * 60 * 60)
    local min = math.floor(time_s / 60)
    time_s = time_s - (min * 60)
    local sec = math.floor(time_s)

    return day, hour, min, sec
end

function SgdMomentum:train(dataset)

    local tstart = os.time()
    local report_every_s = 10
    local report_due_s = report_every_s

    local current_learning_rate = self.learning_rate
    local module = self.module
    local criterion = self.criterion

    local shuffled_indices = torch.randperm(dataset:size(), 'torch.LongTensor')
    if not self.shuffle_indices then
        for t = 1,dataset:size() do
            shuffled_indices[t] = t
        end
    end

    local err_prev = math.huge

    -- Initialize previous weights used to compute momentum.
    criterion:forward(module:forward(dataset[shuffled_indices[1]][1]),
                      dataset[shuffled_indices[1]][2])
    local parameters, gradients = module:parameters()
    local prev_params = {}
    for _, w in ipairs(parameters) do
        table.insert(prev_params, w:clone())
    end

    print("# SgdMomentum: training")

    -- Formatting widths.
    local wid_iter = math.ceil(math.log10(self.max_iteration))
    local wid_points = math.ceil(math.log10(dataset:size())) 
    local fmt_progress = string
        .format("# SgdMomentum: %%02dd%%02dh%%02dm%%02ds : "
                .."iter %%%dd/%%%dd, "
                .."pt %%%dd/%%%dd, "
                .."iter eta %%02dh%%02dm%%02ds, "
                .."eta %%02dd%%02dh%%02dm%%02ds",
                wid_iter, wid_iter, wid_points, wid_points)

    local iteration = 1
    local mini_batch_idx = 1
    local total_point_counter = 0
    while true do

        local current_error = 0
        for t = 1,dataset:size() do
            local example = dataset[shuffled_indices[t]]
            local X = example[1]
            local Y = example[2]
            total_point_counter = total_point_counter + 1

            current_error = current_error + criterion:forward(module:forward(X), Y)
            if current_error ~= current_error
                or current_error == math.huge
                or current_error < 0 then
                error("Error: current_error="..current_error
                      .." after processing "..total_point_counter.." points")
            end
            module:updateGradInput(X, criterion:updateGradInput(module.output, Y))

            -- Add momentum term to gradients.
            local parameters, gradients = module:parameters()
            for i, gw in ipairs(gradients) do
                local w = parameters[i]
                local w_prev = prev_params[i]
                -- -\delta_w = w_{t-1} - w_t
                w_prev:add(-1, w)
                -- Update gradient with momentum term.
                gw:add(-self.momentum / current_learning_rate, w_prev)
                w_prev:copy(w)
            end

            module:accUpdateGradParameters(X, criterion.gradInput, current_learning_rate)

            if self.hookExample then
                self.hookExample(self, example)
            end

            -- Modify learning rate on schedule of mini batches.
            if 0 == (total_point_counter % self.mini_batch_size) then
                mini_batch_idx = mini_batch_idx + 1
                current_learning_rate =
                self.learning_rate / (1 + (mini_batch_idx * self.learning_rate_decay))
            end

            local telapsed = os.time() - tstart
            if telapsed > report_due_s then

                report_due_s = telapsed + report_every_s

                edd, ehh, emm, ess = s_to_ddmmhhss(telapsed)

                local s_per_point = telapsed / total_point_counter

                local remain_this_iter = dataset:size() - t
                local iter_eta = remain_this_iter * s_per_point
                local idd, ihh, imm, iss = s_to_ddmmhhss(iter_eta)
                ihh = ihh + (idd * 24)

                local remain_all_iter = remain_this_iter + 
                    ((self.max_iteration - iteration) * dataset:size())
                local eta = remain_all_iter * s_per_point
                local add, ahh, amm, ass = s_to_ddmmhhss(eta)

                print(string.format(fmt_progress,
                                    edd, ehh, emm , ess,
                                    iteration, self.max_iteration,
                                    t, dataset:size(),
                                    ihh, imm, iss,
                                    add, ahh, amm, ass))
            end
        end

        if self.hookIteration then
            self.hookIteration(self, iteration)
        end

        current_error = current_error / dataset:size()
        print("# current error = "..current_error)

        if self.max_iteration > 0 and iteration >= self.max_iteration then
            print("# SgdMomentum: you have reached the maximum number of iterations")
            break
        end

        -- Check convergence (expect decrease).
        local err_delta = err_prev - current_error
        if err_delta < self.converge_eps then
            print("# SgdMomentum: converged after "..iteration.." iterations")
            break
        end
        err_prev = current_error

        iteration = iteration + 1
    end
end
