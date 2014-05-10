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
  -n, --filename (string) output file name override
  -c, --count (default 1) number of songs to compose
  -p, --play play generated songs
  <INPUT_DIR> (string) directory where input *.mid files reside
  <MODEL_PATH> (string) path to trained model
  <LENGTH> (number) length of generated song
  <OUTPUT_DIR> (string) directory used to save generated song
]]

for key, value in pairs(args) do
    print(key, value)
end

-- Load the model.
model = torch.load(args.MODEL_PATH)
if nil == model then
    error('Could not load model '..args.MODEL_PATH)
end

ds = mid.dataset.load(
        args.INPUT_DIR,
        model.time_sig,
        model.dims.input[2],
        model.dims.output[2],
        0
        )

date_str = os.date("%Y%m%d_%H%M%S")
print('Date string: '..date_str)

-- Get a function that can append a count index to the filename.
local prefix_func
if args.count > 1 then
    prefix_func = function(prefix, n)
        if #prefix > 0 then
            return prefix.."-"..tostring(n)
        else
            return tostring(n)
        end
    end
else
    prefix_func = function(prefix, n)
        return prefix
    end
end

local data = ds.data_test()
local num_points = data:size()

local prefix = args.filename or 'gen-'..date_str
for i = 1, args.count do

    -- Compose a song from a random data point.
    local seed = data[math.random(num_points)][1]
    --local seed = torch.randn(ds.points[1][1]:size()):mul(1e-2)
    local song_data = models.predict(model, seed, args.LENGTH)
    local random_source = ds.sources[math.random(#ds.sources)]
    local song = mid.dataset.compose(random_source, song_data, 0)

    -- Write out generated song.
    gen_filename = prefix_func(prefix, i)..".mid"
    gen_output_path = path.join(args.OUTPUT_DIR, gen_filename)

    print("Composing song "..gen_output_path)

    local file = io.open(gen_output_path, 'w')
    mid.data.write(song.middata, file)
    file:close()

    print("Playing song "..gen_output_path)

    if args.play then
        local cmd = "timidity "..gen_output_path..">/dev/null 2>&1"
        print("Running "..cmd)
        local ret = os.execute(cmd)
        if ret ~= 0 then
            error("Got code "..tostring(ret).." playing song")
        end
    end
end
