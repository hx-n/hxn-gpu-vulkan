package hxn.gpu.vulkan;

/** Shaders from hxnsl, for Vulkan. **/
class Shaders {
	#if macro
	/**
		Registers hxnsl's SPIR-V target for `hxn.gpu.vulkan.Shader`. The package's
		extraParams.hxml runs it; without hxnsl it does nothing.
	**/
	public static function register():Void {
		#if hxnsl
		hxnsl.Targets.register(new hxnsl.targets.Spirv(), "hxn.gpu.vulkan.Shader");
		#end
	}
	#elseif hxnsl

	/**
		The descriptor set layouts of a shader's bind groups, set 0 first:
		what `Device.createPipelineLayout` and `Device.createDescriptorSet`
		take. `layout` is the shader's `LAYOUT`.
	**/
	public static function setLayouts(device:Device, layout:Array<Array<hxnsl.Layout.LayoutEntry>>):SetLayouts {
		return device.createSetLayouts([
			for (group in layout)
				{
					bindings: [
						for (entry in group)
							{binding: entry.binding, kind: kind(entry.type), stages: stages(entry.visibility)}
					]
				}
		]);
	}

	/** The descriptor type of an hxnsl binding type. **/
	public static function kind(type:String):DescriptorType {
		return switch (type) {
			case "uniform": DescriptorType.UniformBuffer;
			case "storage" | "read-only-storage": DescriptorType.StorageBuffer;
			case "sampler" | "comparison-sampler": DescriptorType.Sampler;
			case "texture": DescriptorType.SampledImage;
			case "storage-texture": DescriptorType.StorageImage;
			case other: throw 'hxnsl binding type $other has no Vulkan descriptor type';
		}
	}

	/** Vulkan's shader stage bits for WebGPU's: 1 vertex, 2 fragment, 4 compute. **/
	public static function stages(visibility:Int):Int {
		var bits = 0;
		if (visibility & 1 != 0)
			bits |= ShaderStage.Vertex;
		if (visibility & 2 != 0)
			bits |= ShaderStage.Fragment;
		if (visibility & 4 != 0)
			bits |= ShaderStage.Compute;
		return bits;
	}
	#end
}
