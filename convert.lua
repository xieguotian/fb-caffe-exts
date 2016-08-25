require 'nn';
require 'cunn';
require 'cudnn';
require 'torch2caffe/prepnv.lua'
local t2c=require 'torch2caffe.lib'

-- Figure out the path of the model and load it
local path = arg[1]
local basename = paths.basename(path, 't7b')
local ext = path:match("^.+(%..+)$")
local model = nil
if ext == '.t7b' then 
    model = torch.load(path)
elseif ext == '.txt' then
    error('wrong model')
else
    assert(false, "We assume models end in either .t7b or .txt")
end

if model.net then
	model = model.net
end
model2 = model:clone()
model=g_t2c_preprocess(model, opts)

local function check(module, module2,input_dims)
    module:apply(function(m) m:evaluate() end)
    local opts = {
            prototxt = string.format('%s.prototxt', basename),
            caffemodel = string.format('%s.caffemodel', basename),
            inputs={{name="data", input_dims=input_dims}},
    }
    t2c.convert(opts, module)
    t2c.compare(opts, module2)
    return opts
end


check(model, model2, {1,1,66,200})

