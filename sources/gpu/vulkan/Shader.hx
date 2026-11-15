package gpu.vulkan;

#if hxnsl
/**
	An hxnsl shader for Vulkan. It gets hxnsl's `spirv()` and `LAYOUT`, whose
	bind groups are its descriptor sets; `Shaders.setLayouts` makes their
	layouts.
**/
interface Shader extends hxnsl.Shader {}
#end
