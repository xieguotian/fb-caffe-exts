#!/usr/bin/env th
--++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
--!
--! model_test_loss.lua
--!
--! Brief: -- Script to take a trained model file, with all the included
--!           training and data parameters, and make it usable by our
--!           DrivePX. Major component is to save it as a .txt instead of
--!           tensor binary. However, at the time we are also downgrading
--!           models since the DrivePX is stuck on CUDNNv3
--!
--! Author: NVIDIA CORPORATION,
--!         <a href="http://www.nvidia.com">www.nvidia.com</a>
--!         Created 2016-03-18 by Karol Zieba (kzieba@nvidia.com)
--!
--! Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
--!
--++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


require 'nn';
require 'cunn';
require 'cudnn';
require 'paths';
require 'image'
csv = require 'csvigo'
local t2c=require 'torch2caffe.lib'
local trans = require 'torch2caffe.transforms'

--require 'LocalNormalization'
Test={}
g_SLOW=""

-- Figure out the path of the model and load it
local path = arg[1]
local intenpath = arg[2]
local basename = paths.basename(path, 't7b')
local tenbase = paths.basename(intenpath, 't7b')
local ext = path:match("^.+(%..+)$")
local model = nil
if ext == '.t7b' then 
    model1 = torch.load(path)
else
    assert(false, "We assume models end in either .t7b")
end


local function adapt_spatial_dropout(net)
  -- does not support recursive sequential(dropout)
  --print (model)
    for i = 1, #net.modules do
        local c = net:get(i)
        local t = torch.type(c)
        if c.modules then
            adapt_spatial_dropout(c)
        elseif t == 'nn.SpatialDropout' then
            local m = nn.Dropout(c.p, false,true)
            local found = false
            -- Ignore groups
            net.modules[i] = m
            for j = i,1,-1 do
                local block_type = torch.type(net:get(j))
                if block_type == 'nn.SpatialConvolution'
                    or block_type == 'nn.Linear' then
                 --or block_type == 'nn.SpatialBatchNormalization' then
                    net.modules[j].weight:mul(1 - c.p)
                    if net.modules[j].bias then
                        net.modules[j].bias:mul(1 - c.p)
                    end
                    found = true
                    break
                end
            end
            if not found then
                error('SpatialDropout module cannot find weight to scale')
            end 
        end
    end
end


local function g_t2c_preprocess(model,opts)
  model = cudnn.convert(model, nn)
  model = nn.utils.recursiveType(model, 'torch.FloatTensor')
  for _, layer in pairs(model:findModules('nn.SpatialBatchNormalization')) do
    if layer.save_mean==nil then
      layer.save_mean = layer.running_mean
      layer.save_std = layer.running_var
      layer.save_std:pow(-0.5)
    end
    --layer.train = true
  end
  adapt_spatial_dropout(model)
  return model
end

-- nChannels,nFrames,nGPU,frameInterval,patchHeight,patchWidth,roiWidth,roiVerticalOffset,
-- roiWidthMeters,roiCenterX,baseClamp,supervisor,supervisorNorm

local n_frames = model1.parameters.nFrames
local n_channels = model1.parameters.nChannels
local nGPU = model1.parameters.n_gpu
local frameInterval = model1.parameters.frame_interval
local patch_height = model1.parameters.patch_height
local patch_width = model1.parameters.patch_width
local roiWidth = model1.parameters.roi_width
local roiVerticalOffset = model1.parameters.roi_vertical_offset
local roiWidthMeters = model1.parameters.roi_width_m
local roiCenterX = model1.parameters.roi_center_x
local baseClamp = string.format('\'%s\'', paths.basename(model1.parameters.base_clamp))
local supervisor = string.format('\'%s\'', model1.parameters.supervisor[1])
local supervisorNorm = model1.parameters.supervisor_norms.one_over_r 

csv_string = string.format('nChannels,nFrames,nGPU,frameInterval,patchHeight,patchWidth,roiWidth,roiVerticalOffset,roiWidthMeters,roiCenterX,baseClamp,supervisor,supervisorNorm')

csvf = csv.File(string.format('%s-model-params.csv', basename), "w")
csvf:write({
'nChannels',
'nFrames',
'nGPU',
'frameInterval',
'patchHeight',
'patchWidth',
'roiWidth',
'roiVerticalOffset',
'roiWidthMeters',
'roiCenterX',
'baseClamp',
'supervisor',
'supervisorNorm'})

csvf:write({
n_channels,
n_frames,
nGPU,
frameInterval,
patch_height,
patch_width,
roiWidth,
roiVerticalOffset,
roiWidthMeters,
roiCenterX,
baseClamp,
supervisor,
supervisorNorm})
csvf:close()

if model1.net then
  model1 = model1.net
end

model1 = g_t2c_preprocess(model1, opts)

local function check(net, input_dims)
    net:apply(function(m) m:evaluate() end)
    local opts = {
            prototxt = string.format('%s.prototxt', basename),
            caffemodel = string.format('%s.caffemodel', basename),
            inputs = {{name = "data", input_dims = input_dims, tensor = torch.load(intenpath)[1]:view(1, 1, 66, 200)}}}  
    t2c.compare(opts, net)
    return opts
end

check(model1, {1, n_frames * n_channels, patch_height, patch_width})
--
testpatch = torch.load(intenpath)
testpatch = testpatch[1]:clone()
torch.save(string.format("%s.t7b",tenbase), testpatch)
image.save(string.format("%s.JPEG",tenbase), image.toDisplayTensor(testpatch))

