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
require 'Rnn'

--[
-- TODO:
--
--   REGRESSION
--   a) Create simple deep network predicting next notes across channels as a
--      form of regression. This simplifies dealing with softmax and class
--      labels on the output, but in general it may be more difficult to get
--      reasonable performance.
--   b) Experiment with the loss function on the regression problem.
--
--   MULTICLASS
--   a) Use K classes to represent note velocity obtained by median filtering
--      the velocity values observed in the training data. Applying softmax
--      to the output gives a simple loss function.
--
--   CONVNET
--   a) Try several deep convnet architectures.
--
--   For all models, experiment with input dimensions (window size) and also
--   output dimensions.
--
--   RECURRENT
--   a) Experiment with the unrolling length.
--]

local args = lapp [[
Train a learning machine to predict sequences of notes extracted from MIDI
files.
  -i, --input-window-size (default 10) size in gcd ticks of input point X
  -o, --output-window-size (default 1) size in gcd ticks of target point Y
  -r, --rnn is a recurrent neural network; note that output window must be > 1
  -s, --dataset-train-split (default 0.9) percentage of data to use for training
  -h, --hidden-units (default 256) number of 2lnn hidden units
  -t, --model-type (default "iso") model type in {iso, iso+cmb, cmb}
                   iso - isolated, features apply to a single note
                   cmb - combined, features apply across notes
                   iso+cmb - isolated features combined across notes
  <INPUT_DIR> (string) directory where input *.mid files reside
  <TIME_SIG_CHANNELS_GCD> (string) time signature, channels, and gcd
                          e.g. 4/2-8-24-4-256
  <OUTPUT_DIR> (string) directory used to save output model file
               and an example generated song
]]

local ds
if args.rnn then
    ds = mid.dataset.load_rnn(
            args.INPUT_DIR,
            args.TIME_SIG_CHANNELS_GCD, 
            args.input_window_size,
            args.output_window_size,
            args.dataset_train_split
            )
else
    ds = mid.dataset.load(
            args.INPUT_DIR,
            args.TIME_SIG_CHANNELS_GCD, 
            args.input_window_size,
            args.output_window_size,
            args.dataset_train_split
            )
end

-- Show command-line options.
for key, value in pairs(args) do
    print(key, value)
end

-- Generate date string used to label model and test midi.
date_str = os.date("%Y%m%d_%H%M%S")
print('Date string: '..date_str)
print('Num training: '..ds.data_train():size())

-- Create 2-layer NN with specified hidden units. Each hidden unit is a feature
-- extractor that is applied to an input time slice for a single note.
local model
if "iso" == args.model_type then
    model = models.simple_2lnn_iso(ds, args.hidden_units)
elseif "cmb" == args.model_type then
    model = models.simple_2lnn_cmb(ds, args.hidden_units)
elseif "iso+cmb" == args.model_type then
    model = models.simple_2lnn_iso_cmb(ds, args.hidden_units)
end

-- Create RNN when requested.
local train_model
local model_type
if args.rnn then
    train_model = nn.Rnn(model)
    model_type = "rnn-"..args.model_type
else
    train_model = model
    model_type = "2lnn-"..args.model_type
end

local to_byte = mid.dataset.default_to_byte
local criterion = nn.MSECriterion()
--local criterion = nn.AbsCriterion()
--local criterion = nn.PerceptualLoss(to_byte)

-- Initilaize weights but first run newtork to size Tensors.
criterion:forward(train_model:forward(ds.points[1][1]), ds.points[1][2])
train_model:reset(1e-1)

-- Train.
err_train, err_test = models.train_model(ds, train_model, criterion)
print("avg error train/test", err_train, err_test)

-- Append command-line options to the model.
for key, value in pairs(args) do
    model[key] = value
end

-- Write out the model.
model_filename = 'model-'..model_type..args.hidden_units..'-'..date_str
model_output_path = path.join(args.OUTPUT_DIR, model_filename)
torch.save(model_output_path, model)
print("Wrote model "..model_output_path)

-- Generate a song.
song_data = models.predict(model,
                           ds.data_test()[1][1]:narrow(2, 1, args.input_window_size),
                           10)
song = mid.dataset.compose(ds.sources[1], song_data, 4)

-- Write out generated song.
gen_filename = 'gen-'..date_str..'.mid'
gen_output_path = path.join(args.OUTPUT_DIR, gen_filename)
mid.data.write(song.middata, io.open(gen_output_path, 'w'))
print("Wrote song "..gen_output_path)
