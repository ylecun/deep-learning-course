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

-- Get lua module path.
local bin_dir = path.dirname(arg[0])
local lua_dir = path.abspath(path.join(bin_dir, "../lua"))
local mod_pattern = path.join(lua_dir, "?", "init.lua")
local script_pattern = path.join(lua_dir, "?.lua")
package.path = mod_pattern..";"..script_pattern..";"..package.path

require 'mid'
require 'models'
require 'nn'

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
  -s, --dataset-train-split (default 0.8) percentage of data to use for training
  <INPUT_DIR> (string) directory where input *.mid files reside
  <TIME_SIG_CHANNELS_GCD> (string) time signature, channels, and gcd
                          e.g. 4/2-8-24-5-256
  <OUTPUT_DIR> (string) directory used to save output model file
               and an example generated song
]]

for key, value in pairs(args) do
    print(key, value)
end

ds = mid.dataset.load(
        args.INPUT_DIR,
        args.TIME_SIG_CHANNELS_GCD, 
        args.input_window_size,
        args.output_window_size,
        args.dataset_train_split
        )

date_str = os.date("%Y%m%d_%H%M%S")

HIDDEN_UNITS = 256

-- Create 2-layer NN with HIDDEN_UNITS hidden units. Each hidden unit is a
-- feature extractor that is applied to an input time slice for a single note.
model = models.simple_2lnn(ds, HIDDEN_UNITS)
-- Train with simple regression loss.
-- TODO: a more perceptual loss function is ideal
models.train_model(ds, model, nn.MSECriterion())

-- Write out the model.
model_filename = 'model-2lnn-'..HIDDEN_UNITS..'-'..date_str
model_output_path = path.join(args.OUTPUT_DIR, model_filename)
torch.save(model_output_path, model)

song_data = models.predict(model, ds.data_test()[1][1], 10)
song = dataset.compose(ds.sources[1], song_data)

-- Write out generated song.
gen_filename = 'gen-'..date_str..'.mid'
gen_output_path = path.join(args.OUTPUT_DIR, gen_filename)
mid.data.write(song.middata, io.open(gen_output_path, 'w'))
