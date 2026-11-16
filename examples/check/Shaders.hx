/** Each value becomes three times itself plus one. **/
class Tripled implements hxn.gpu.vulkan.Shader {
	static var SRC = {
		@param var data : RWBuffer<Int>;

		function main() {
			setLayout(16);
			var i = computeVar.globalInvocation.x;
			data[i] = data[i] * 3 + 1;
		}
	};
}

/** Fills what it covers with `tint`. **/
class Tinted implements hxn.gpu.vulkan.Shader {
	static var SRC = {
		@input var input : { position : Vec2 };
		var output : { position : Vec4, color : Vec4 };
		@param var tint : Vec4;

		function vertex() {
			output.position = vec4(input.position, 0, 1);
		}

		function fragment() {
			output.color = tint;
		}
	};
}
