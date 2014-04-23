--[
-- Compose a new song given a model.
--
-- (c) 2014 Brandon L. Reiss
--]
lapp = require 'pl.lapp'
path = require 'pl.path'
require "os"
require "paths"
require "math"

-- Get lua module path.
local bin_dir = paths.dirname(paths.thisfile())
local lua_dir = path.abspath(path.join(bin_dir, "../lua"))
local mod_pattern = path.join(lua_dir, "?", "init.lua")
local script_pattern = path.join(lua_dir, "?.lua")
package.path = mod_pattern..";"..script_pattern..";"..package.path

require 'mid'
require 'models'
require 'nn'

local args = lapp [[
Compose a new song from a model.
  -i, --input-window-size (default 10) size in gcd ticks of input point X
  -o, --output-window-size (default 1) size in gcd ticks of target point Y
  -n, --filename (string) output file name override
  <INPUT_DIR> (string) directory where input *.mid files reside
  <TIME_SIG_CHANNELS_GCD> (string) time signature, channels, and gcd
                          e.g. 4/2-8-24-4-256
  <MODEL_PATH> (string) path to trained model
  <LENGTH> (number) length of generated song
  <OUTPUT_DIR> (string) directory used to save generated song
]]

for key, value in pairs(args) do
    print(key, value)
end

ds = mid.dataset.load(
        args.INPUT_DIR,
        args.TIME_SIG_CHANNELS_GCD, 
        args.input_window_size,
        args.output_window_size,
        0
        )

date_str = os.date("%Y%m%d_%H%M%S")
print('Date string: '..date_str)

-- Load the model.
model = torch.load(args.MODEL_PATH)

if nil == model then
    error('Could not load model '..args.MODEL_PATH)
end

-- Compose a song from a random data point.
local data = ds.data_test()
local num_points = data:size()
local random_point = math.random(num_points)
local seed = data[random_point][1]
song_data = models.predict(model, seed, args.LENGTH)
song = mid.dataset.compose(ds.sources[1], song_data, 4)

-- Write out generated song.
gen_filename = args.filename or 'gen-'..date_str..'.mid'
gen_output_path = path.join(args.OUTPUT_DIR, gen_filename)
mid.data.write(song.middata, io.open(gen_output_path, 'w'))
