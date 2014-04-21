--[
-- Main command-line application for ingesting and training JSB Chorales MIDI
-- files and then building a learning machine that attempts to produce
-- Bach-like music given a sequence of initial notes.
--
-- (c) 2014 Brandon L. Reiss
--]
lapp = require 'pl.lapp'
path = require 'pl.path'
require "os"
require "paths"

-- Get lua module path.
local bin_dir = paths.dirname(paths.thisfile())
local lua_dir = path.abspath(path.join(bin_dir, "../lua"))
local mod_pattern = path.join(lua_dir, "?", "init.lua")
local script_pattern = path.join(lua_dir, "?.lua")
package.path = mod_pattern..";"..script_pattern..";"..package.path

require 'mid'
require 'models'
require 'nn'
require 'PerceptualLoss'

--[
-- TODO:
--
--   REGRESSION
--   a) Create simple deep network predicting next notes across channels as a
--      form of regression. This simplifies dealing with softmax and class
--      labels on the output, but in general it may be more difficult to get
--      reasonable performance.
--   b) Try a random switch inside of the network as a way to create an
--      ensemble that does not have repeating dynamics or enless loops.
--   c) Experiment with the loss function on the regression problem.
--
--   MULTICLASS
--   a) Use K classes to represent note velocity obtained by median filtering
--      the velocity values observed in the training data. Applying softmax
--      to the output gives a simple loss function.
--
--   For either model, experiment with input dimensions (window size) and also
--   output dimensions.
--
--   RECURRENT
--   How do we make one of these?
--
--   CONVERT
--   Need to convert rasterized data back to MIDI and test for known input.
--   Need to decide how to setup tracks and instruments for predicted outputs.
--]

local args = lapp [[
Train a learning machine to predict sequences of notes extracted from MIDI
files.
  -i, --input-window-size (default 10) size in gcd ticks of input point X
  -o, --output-window-size (default 1) size in gcd ticks of target point Y
  -s, --dataset-train-split (default 0.9) percentage of data to use for training
  -h, --hidden-units (default 256) number of 2lnn hidden units
  <INPUT_DIR> (string) directory where input *.mid files reside
  <TIME_SIG_CHANNELS_GCD> (string) time signature, channels, and gcd
                          e.g. 4/2-8-24-4-256
  <OUTPUT_DIR> (string) directory used to save output model file
               and an example generated song
]]

ds = mid.dataset.load(
        args.INPUT_DIR,
        args.TIME_SIG_CHANNELS_GCD, 
        args.input_window_size,
        args.output_window_size,
        args.dataset_train_split
        )

--print("SELECTING IDX 562")
--for i = 1, #ds.points do
--    local i_wnd = ds.points[i][1]:size(2)
--    local o_wnd = ds.points[i][2]:size(2)
--    ds.points[i][1] = ds.points[i][1][562]:reshape(1, i_wnd)
--    ds.points[i][2] = ds.points[i][2][562]:reshape(1, o_wnd)
--end

for key, value in pairs(args) do
    print(key, value)
end

date_str = os.date("%Y%m%d_%H%M%S")
print('Date string: '..date_str)
print('Num training: '..ds.data_train():size())

-- Create 2-layer NN with specified hidden units. Each hidden unit is a feature
-- extractor that is applied to an input time slice for a single note.
model = models.simple_2lnn(ds, args.hidden_units)

--models.train_model(ds, model, nn.AbsCriterion())
err_train, err_test = models.train_model(ds, model, PerceptualLoss())

print("avg error train/test", err_train, err_test)

-- Write out the model.
model_filename = 'model-2lnn-'..args.hidden_units..'-'..date_str
model_output_path = path.join(args.OUTPUT_DIR, model_filename)
torch.save(model_output_path, model)

song_data = models.predict(model, ds.data_test()[1][1], 10)
song = mid.dataset.compose(ds.sources[1], song_data, 4)

-- Write out generated song.
gen_filename = 'gen-'..date_str..'.mid'
gen_output_path = path.join(args.OUTPUT_DIR, gen_filename)
mid.data.write(song.middata, io.open(gen_output_path, 'w'))
