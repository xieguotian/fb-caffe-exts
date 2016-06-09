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
--require 'LocalNormalization'

-- Figure out the path of the model and load it
local path = arg[1]
local ext = path:match("^.+(%..+)$")
local model = nil
if ext == '.t7b' then 
  model = torch.load(path)
elseif ext == '.txt' then
  model = torch.load(path, 'ascii')
else
  assert(false, "We assume models end in either .t7b or .txt")
end

-- Just incase we perform any calculations for the future.
torch.setdefaulttensortype(model.parameters.tensor_type)

-- Batch normalization is implemented in CUDNNv4 differently than in CUDNNv3 so
-- we convert models to the NN implementation recursively and in place.
-- Also nn.SBN has extra table parameters which crash the system so we remove them
function convert_bn(net)
  for i = 1, #net.modules do
    local c = net:get(i)
    local t = torch.type(c)
    if c.modules then
      convert_bn(c)
    elseif t == 'cudnn.SpatialBatchNormalization' then
      local m = nn.SpatialBatchNormalization(c.weight:size(1)):cuda()
      m.gradBias = c.gradBias
      m.gradInput = c.gradInput
      m.gradWeight = c.gradWeight
      m.output = c.output
      if c.running_std == nil then
        m.running_var = c.running_var
      else -- handle the case of CUDNN 4 and before using 1/std instead of var
        m.running_var = c.running_std
        m.running_var:pow(-2)
      end 
      m.running_mean = c.running_mean
      m.weight = c.weight
      m.bias = c.bias
      m.train = c.train
      m.affine = c.affine
      m.save_std = nil
      m.save_mean = nil
      net.modules[i] = m
    elseif t == 'nn.SpatialBatchNormalization' then
      c.save_std = nil
      c.save_mean = nil
    end
  end
end
function convert_conv(net)
  for i = 1, #net.modules do
    local c = net:get(i)
    local t = torch.type(c)
    if c.modules then
      convert_conv(c)
    elseif t == 'cudnn.SpatialConvolution' then
      local m = nn.SpatialConvolution(c.nInputPlane, c.nOutputPlane, c.kW, c.kH,
                                      c.dW, c.dH, c.padW, c.padH)
      m.output = c.output
      m.gradInput = c.gradInput
      m._type = c._type
      m.weight = c.weight
      m.gradWeight = c.gradWeight
      m.bias = c.bias
      m.gradBias = c.gradBias
      -- Ignore groups
      net.modules[i] = m
    end
  end
end
  

print(model.net)
convert_bn(model.net)
convert_conv(model.net)
print(model.net)


-- Save the model
local out_path = path:sub(1, #path - 4) .. "_clean.t7b"
torch.save(out_path, model.net, 'binary', false)
