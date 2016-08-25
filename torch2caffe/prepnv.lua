require 'nn'
require 'cunn'
require 'cudnn'

local trans = require 'torch2caffe.transforms'

local function adapt_conv1(layer)
  local std = torch.FloatTensor({0.229, 0.224, 0.225}) * 255
  local sz = layer.weight:size()
  sz[2] = 1
  layer.weight = layer.weight:cdiv(std:view(1,3,1,1):repeatTensor(sz))
  local tmp = layer.weight:clone()
  tmp[{{}, 1, {}, {}}] = layer.weight[{{}, 3, {}, {}}]
  tmp[{{}, 3, {}, {}}] = layer.weight[{{}, 1, {}, {}}]
  layer.weight = tmp:clone()
end

local function adapt_spatial_dropout(net)
  --print (model)
  for i = 1, #net.modules do
    local c = net:get(i)
    local t = torch.type(c)
		if c == nil then
			break
		end
    if c.modules then
      adapt_spatial_dropout(c)
    elseif t == 'nn.SpatialDropout' then
      local found = false
			-- find the previous layer and scale
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
			for j = i, net:size()-1 do
				net.modules[j] = net.modules[j + 1]
			end
				net.modules[net:size()] = nil
	  end
  end
end

remove_flatten = function(net)
  for i = 1, #net.modules do
    local c = net:get(i)
    local t = torch.type(c)
    if c.modules then
      remove_flatten(c)
    elseif t == 'nn.Reshape' then
      print('Flatten layer is founded!')
      for j = i, #net.modules-1  do
        net.modules[j] = net.modules[j+1]
      end
      net.modules[#net.modules] = nil
      break
    end
  end
end

g_t2c_preprocess = function(model, opts)
		-- convert the model to cpu mode
    if model.net then
        model = model.net
    end
    model = cudnn.convert(model, nn)
    model=nn.utils.recursiveType(model, 'torch.FloatTensor')

    for _, layer in pairs(model:findModules('nn.SpatialBatchNormalization')) do
        if layer.save_mean==nil then
            layer.save_mean = layer.running_mean
            layer.save_std = layer.running_var
            layer.save_std:pow(-0.5)
        end
        --layer.train = true
    end
    --adapt_spatial_dropout(model)
  	remove_flatten(model)
    return model
end

save_model_params = function(model, basename)
	-- saving the model-parameters
	local n_frames = model.parameters.nFrames
	local n_channels = model.parameters.nChannels
	local nGPU = model.parameters.n_gpu
	local frameInterval = model.parameters.frame_interval
	local patch_height = model.parameters.patch_height
	local patch_width = model.parameters.patch_width
	local roiWidth = model.parameters.roi_width
	local roiVerticalOffset = model.parameters.roi_vertical_offset
	local roiWidthMeters = model.parameters.roi_width_m
	local roiCenterX = model.parameters.roi_center_x
	local targetClamp = string.format('\'%s\'', paths.basename(model.parameters.target_clamp))
	local supervisor = string.format('\'%s\'', model.parameters.supervisor[1])
	local supervisorNorm = model.parameters.supervisor_norms.one_over_r 

	local csvf = csv.File(string.format('%s-model-params.csv', basename), "w")
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
		targetClamp,
		supervisor,
		supervisorNorm})
		csvf:close()
end
