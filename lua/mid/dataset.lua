--[
-- Tools to represent a .mid file as a Torch dataset.
--
-- (c) 2014 Brandon L. Reiss
--]
require "torch"
require "io"
require "math"

require "util"
local mid_data = require "mid/data"
local mid_event = require "mid/event"

local CODES = mid_event.CODES

local binary_search = util.binary_search

--- Number of unique note values allowed by MIDI.
local NOTE_DIMS = 128

dataset = {}

--- Compute the gcd of either
--    1) a pair of values (a, b) or
--    2) a list of values (a, b, c, ...)
local function gcd(a, b)

    -- Support either (a, b) or {a, b, c, ...}.
    local values
    if "table" == type(a) then
        values = a
    else
        values = {a, b}
    end
    if #values < 2 then
        print(values)
        error("Must have at least 2 values for gcd")
    end

    --- Helper for gcd using Euclid's algorithm.
    local function gcd_pair(a, b)
        if b == 0 then
            return a
        else
            return gcd_pair(b, a % b)
        end
    end

    -- The gcd of the list is the gcd of any subsequence of the list and the
    -- next element.
    local gcd = gcd_pair(values[1], values[2])
    for i = 2,(#values - 1) do
        gcd = gcd_pair(gcd, values[i])
    end

    return gcd
end

--- Create a hash code for a time signature event. We need to separate
-- learning machines by the timing of their music.
local function hash_time_signature_event(event)
    -- Only delta time of 0 supported.
    if 0 ~= event.delta_time then
        error("Time signature set at delta != 0")
    end

    return string.format(
            "%d/%d-%d-%d",
            event.numerator,
            event.denominator,
            event.n32_notes_per_qrtr,
            event.metro_ticks_per_click)
end

--- Create a discretization of the midi file such that each tensor row is a
-- simple accounting of the notes that are on and off separated by channel.
-- The notes are codes from 0-127.
--
-- The tensor for the entire piece will be
--     ((max_clock - min_clock) / clock_gcd) x (channels * NOTE_DIMS)
--
-- We require hashing events by note and then sorting them by time.  The
-- entire sequence length is known, so we create an empty bitmap and then
-- fill in the notes where they occur.
local function rasterize_song(channels, min_clock, max_clock, clock_gcd, note_events)

    local raster_clock_offset = -min_clock / clock_gcd
    local time_steps_to_render = (max_clock / clock_gcd) + raster_clock_offset
    local channel_note_dims = #channels * NOTE_DIMS
    local rasterized = torch.zeros(time_steps_to_render, channel_note_dims)
    --print(rasterized:size())

    for note, by_channel in pairs(note_events) do

        for channel, clock_events in pairs(by_channel) do
            local channel_idx = binary_search(channels, channel)
            local column_idx = ((channel_idx - 1) * NOTE_DIMS) + note + 1
            local note_channel_data = rasterized:narrow(2, column_idx, 1)

            local note_begin, velocity = nil, nil

            for _, clock_event in ipairs(clock_events) do
                local raster_clock = (clock_event.clock / clock_gcd) + raster_clock_offset + 1
                local event = clock_event.e

                if event.midi == CODES.midi.note_on then
                    note_begin = raster_clock
                    velocity = event.velocity / 255.0

                elseif event.midi == CODES.midi.note_off
                    and note_begin ~= nil then
                    -- Set range for this note on with velocity.
                    local duration = raster_clock - note_begin
                    note_channel_data:narrow(1, note_begin, duration):fill(velocity)
                    note_begin, velocity = nil, nil
                end

            end
        end
    end

    return rasterized
end

--- Load a .mid file as a torch dataset.
--@param dir the directory containing .mid files
--@param time_sig the time signature in the form
--    num/denom-32notesperquarter-ticksperclick-channels-gcd
--@param input_len the number of discrete time steps per input point X
--@param target_len the number of discrete time steps per target point Y
--@param pct_train the proportion of points to use as training data
--@return train and test sets of pairs (X, Y)
function dataset.load(dir, time_sig, input_len, target_len, pct_train)

    -- Get all midi files in path.
    mid_files = {}
    for filename in lfs.dir(dir) do
        if filename:lower():find(".mid") ~= nil then
            table.insert(mid_files, filename)
        end
    end

    -- Collect rasterized data matching time signature.
    sources = {}

    -- For each file.
    for _,filename in ipairs(mid_files) do

        local file_path = dir.."/"..filename
        local file = io.open(file_path, "rb")

        local status, err = pcall(function()

            -- Load MIDI data.
            middata = mid_data.read(file)

            -- We are going to find
            --   1) a hash of the time signature
            --   2) the gcd of all note event time deltas
            --   3) the (min, max) clock for any note
            --   4) a dictionary note events { note -> channel -> {events} }
            local note_events = (function()
                local t = {}
                for i = 1,NOTE_DIMS do
                    t[i] = {}
                end
                return t
            end)()
            local channels_on = {}
            local time_hash = nil
            local min_clock, max_clock = math.huge, -1
            local clock_gcd = 0

            -- Loop over all tracks and events.
            for _, track in ipairs(middata.tracks) do
                local clock = 0
                for _, event in ipairs(track.events) do

                    clock = clock + event.delta_time

                    if event.midi == CODES.midi.note_on or
                        event.midi == CODES.midi.note_off then

                        -- Get channel-specific list of events for this note.
                        channels_on[event.channel] = true
                        local by_channel = note_events[event.note_number]
                        if by_channel[event.channel] == nil then
                            by_channel[event.channel] = {}
                        end
                        local events = by_channel[event.channel]
                            
                        -- Wrap events as a clock event with absolute time.
                        table.insert(events,
                                {
                                    clock = clock,
                                    e = event
                                })

                        clock_gcd = gcd(clock_gcd, clock)
                        min_clock = math.min(min_clock, clock)
                        max_clock = math.max(max_clock, clock)

                    elseif event.meta == CODES.meta.time_sig then
                        time_hash = hash_time_signature_event(event)
                    end
                end
            end

            -- Get sorted list of channels from table of active channels.
            local channels = (function()
                local channels = {}
                for channel in pairs(channels_on) do
                    table.insert(channels, channel)
                end
                table.sort(channels)
                return channels
            end)()

            local my_time_sig = time_hash.."-"..#channels.."-"..clock_gcd

            -- Check that time signature matches filter type.
            if my_time_sig == time_sig then

              -- Rasterize song and append to output.
              local rasterized = rasterize_song(
                      channels, min_clock, max_clock, clock_gcd, note_events)

              --print(file_path)
              --print("time sig: "..my_time_sig)
              --print("min, max note clock: "..min_clock..","..max_clock)
              --print("dims: "..tostring(rasterized:size()))

              table.insert(sources, {
                  name = filename,
                  data = rasterized,
                  middata = middata,
              })
            end

            --print(rasterized:size())

        end)

        file:close()
    end

    -- For each valid source, split data into points and then partition into
    -- train and test sets. Each torch narrow() does not copy data so it can
    -- store efficiently overlapping subsequences of the source.
    local points = {}
    local input_target_len = input_len + target_len
    for _, source in ipairs(sources) do
        local data = source.data
        for i = input_target_len,data:size()[1] do
            local X_begin = i - input_target_len + 1
            local Y_begin = i - target_len + 1
            local X, Y = data:narrow(1, X_begin, input_len),
                data:narrow(1, Y_begin, target_len)
            table.insert(points, {X, Y})
        end
    end

    -- Shuffle the points and then partition into train, test sets.
    util.shuffle(points)
    local num_train = math.ceil(#points * pct_train)
    local num_test = #points - num_train

    --- Train dataset iterator.
    local function data_train()
        local i = 0
        return function()
            i = i + 1
            if i <= num_train then
                return points[i]
            else
                return nil
            end
        end
    end

    --- Test dataset iterator.
    local function data_test()
        local i = num_train
        return function()
            i = i + 1
            if i <= #points then
                return points[i]
            else
                return nil
            end
        end
    end

    return {
        data_train = data_train,
        data_test = data_test(),
        num_train = num_train,
        num_test = num_test,
        points = points,
        sources = sources,
    }
end

return dataset
