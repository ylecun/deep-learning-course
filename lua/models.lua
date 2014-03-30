--[
-- NN package model helpers.
--
-- (c) 2014 Brandon L. Reiss
--]
require "nn"
require "math"
require "torch"

models = {}

--- Create a simple 2-layer NN.
function models.simple_2lnn(ds, num_hidden)

    local train = ds.data_train()
    local input_wnd = train[1][1]:size(2)
    local output_wnd = train[1][2]:size(2)

    local mlp = nn.Sequential()
    mlp:add(nn.Linear(input_wnd, num_hidden))
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(num_hidden, output_wnd))

    return mlp
end

--- Train model using the given dataset.
function models.train_model(ds, model, criterion)

    local train = ds.data_train()
    local trainer = nn.StochasticGradient(model, criterion)
    trainer:train(train)

    local train_loss = 0
    for i = 1, train:size() do
        X = train[i][1]
        Y = train[i][2]
        loss = criterion:forward(model:forward(X), Y)
        train_loss = train_loss + loss
    end
    local avg_train_loss = train_loss / train:size()

    local test_loss = 0
    local test = ds.data_test()
    for i = 1, test:size() do
        local X = test[i][1]
        local Y = test[i][2]
        local loss = criterion:forward(model:forward(X), Y)
        test_loss = test_loss + loss
    end
    local avg_test_loss = test_loss / test:size()

    return avg_train_loss, avg_test_loss
end

--- Predict a song by seeding with input window x0.
-- :param number length: the length of the song in output windows
function models.predict(model, x0, length)

    local input_wnd = x0:size(2)
    local output_wnd =  model:forward(x0):size(2)

    local channel_dims = x0:size(1)
    local x0_song = torch.Tensor(channel_dims, input_wnd + (output_wnd * length))
    x0_song:narrow(2, 1, input_wnd):copy(x0)
    local song = x0_song:narrow(2, input_wnd, output_wnd * length)

    for i = 1, length do
        -- Predict next output_wnd and copy to the song. 
        offset = 1 + ((i - 1) * output_wnd)
        local X = x0_song:narrow(2, offset, input_wnd)
        local Y = model:forward(X)
        song:narrow(2, offset, output_wnd):copy(Y)
    end

    return song

end

return models
