require 'nn';
require 'cunn';
require 'cudnn';
require 'paths';
require 'image'
require 'torch2caffe/prepnv.lua'
local t2c=require 'torch2caffe.lib'


-- Figure out the path of the model and load it
local path = arg[1]
local intenpath = arg[2]
local basename = paths.basename(path, 't7b')
local ext = path:match("^.+(%..+)$")
local model = nil
if ext == '.t7b' then 
    model1 = torch.load(path)
else
    assert(false, "We assume models end in either .t7b")
end

if model1.net then
	model1 = model1.net
end

local function check_input(net, input_dims, input_tensor)
    net:apply(function(m) m:evaluate() end)
    local opts = {
            prototxt = string.format('%s.prototxt', basename),
            caffemodel = string.format('%s.caffemodel', basename),
            inputs = {{
				name = "data", 
				input_dims = input_dims, 
				tensor = input_tensor
				}}
			}  
    t2c.compare(opts, net)
    return opts
end

local function check(net, input_dims)
    net:apply(function(m) m:evaluate() end)
    local opts = {
            prototxt = string.format('%s.prototxt', basename),
            caffemodel = string.format('%s.caffemodel', basename),
            inputs = {{
				name = "data", 
				input_dims = input_dims, 
				}}
			}  
    t2c.compare(opts, net)
    return opts
end

if intenpath ~= nil then
	print('Using given input tensor', intenpath)
	input = torch.load(intenpath):view(table.unpack(input_dims))
	check_input(model1, {1, n_channels, patch_height, patch_width}, input)
	--
	torch.save(string.format("%s.t7b", input), testpatch)
	image.save(string.format("%s.JPEG", input), image.toDisplayTensor(testpatch))
else
	print('Creating new tensor')
	input = nil
	check(model1, {1, 1, 66, 200})
end


