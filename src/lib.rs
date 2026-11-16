//! Vulkan for Haxe: a thin layer over Vulkan, keeping its conventions. Shaders are hxnsl's
//! SPIR-V, whose descriptor sets are hxnsl's bind groups. Synchronization is explicit: a
//! program records the barriers and image layout transitions its commands need, as in Vulkan.
//!
//! Rendering uses dynamic rendering and barriers use synchronization2, from Vulkan 1.3 or
//! their extensions on Vulkan 1.2, which is what MoltenVK provides.

use std::ffi::{CStr, CString, c_char};
use std::sync::{Arc, Mutex};

use ash::vk;
use hxn_future::Future;

hxn::package!("hxn.gpu.vulkan");

fn failed(what: &str, result: vk::Result) -> String {
    format!("{what}: {result}")
}

fn usize_of(value: i64, what: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{what} cannot be {value}"))
}

fn u32_of(value: i32, what: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{what} cannot be {value}"))
}

/// The loader and the `VkInstance`, destroyed after every device made from it.
struct InstanceShared {
    _entry: ash::Entry,
    instance: ash::Instance,
}

impl Drop for InstanceShared {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// A `VkDevice` and what its objects need of it, destroyed after every one of them.
struct DeviceShared {
    instance: Arc<InstanceShared>,
    device: ash::Device,
    // Submits are externally synchronized in Vulkan.
    queue: Mutex<vk::Queue>,
    family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    sync2: ash::khr::synchronization2::Device,
    rendering: ash::khr::dynamic_rendering::Device,
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
        let _ = &self.instance;
    }
}

impl DeviceShared {
    fn memory_type(&self, bits: u32, wanted: vk::MemoryPropertyFlags) -> Result<u32, String> {
        (0..self.memory.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && self.memory.memory_types[i as usize]
                        .property_flags
                        .contains(wanted)
            })
            .ok_or_else(|| format!("the device has no memory that is {wanted:?}"))
    }
}

/// The formats an image or vertex attribute can have, with Vulkan's values.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum Format {
    R8Unorm = 9,
    Rgba8Unorm = 37,
    Rgba8Srgb = 43,
    Bgra8Unorm = 44,
    Bgra8Srgb = 50,
    Rgba16Float = 97,
    R32Uint = 98,
    R32Sint = 99,
    R32Float = 100,
    Rg32Float = 103,
    Rgb32Float = 106,
    Rgba32Float = 109,
    D32Float = 126,
}

impl Format {
    fn raw(self) -> vk::Format {
        vk::Format::from_raw(self as i32)
    }
}

/// `Device.createBuffer` usage bits, Vulkan's values.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum BufferUsage {
    TransferSrc = 1,
    TransferDst = 2,
    Uniform = 16,
    Storage = 32,
    Index = 64,
    Vertex = 128,
}

/// `ImageDescriptor.usage` bits, Vulkan's values.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum ImageUsage {
    TransferSrc = 1,
    TransferDst = 2,
    Sampled = 4,
    Storage = 8,
    ColorAttachment = 16,
    DepthAttachment = 32,
}

#[hxn::data]
pub struct ImageDescriptor {
    pub width: i32,
    pub height: i32,
    pub format: Format,
    /// `ImageUsage` bits, or-ed together.
    pub usage: i32,
}

/// An image layout, Vulkan's values.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum Layout {
    Undefined = 0,
    General = 1,
    ColorAttachment = 2,
    DepthAttachment = 3,
    ShaderReadOnly = 5,
    TransferSrc = 6,
    TransferDst = 7,
    Present = 1000001002,
}

impl Layout {
    fn raw(self) -> vk::ImageLayout {
        vk::ImageLayout::from_raw(self as i32)
    }
}

/// Pipeline stage bits for a barrier, Vulkan's values, or-ed together.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum Stage {
    TopOfPipe = 1,
    DrawIndirect = 2,
    VertexInput = 4,
    VertexShader = 8,
    FragmentShader = 128,
    EarlyFragmentTests = 256,
    LateFragmentTests = 512,
    ColorAttachmentOutput = 1024,
    ComputeShader = 2048,
    Transfer = 4096,
    BottomOfPipe = 8192,
    Host = 16384,
    AllGraphics = 32768,
    AllCommands = 65536,
}

/// Memory access bits for a barrier, Vulkan's values, or-ed together.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum Access {
    IndirectCommandRead = 1,
    IndexRead = 2,
    VertexAttributeRead = 4,
    UniformRead = 8,
    ShaderRead = 32,
    ShaderWrite = 64,
    ColorAttachmentRead = 128,
    ColorAttachmentWrite = 256,
    DepthAttachmentRead = 512,
    DepthAttachmentWrite = 1024,
    TransferRead = 2048,
    TransferWrite = 4096,
    HostRead = 8192,
    HostWrite = 16384,
    MemoryRead = 32768,
    MemoryWrite = 65536,
}

fn stages(bits: i32) -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::from_raw(bits as u32 as u64)
}

fn access(bits: i32) -> vk::AccessFlags2 {
    vk::AccessFlags2::from_raw(bits as u32 as u64)
}

/// A descriptor's type, Vulkan's values.
#[hxn::data]
#[derive(Clone, Copy, PartialEq)]
pub enum DescriptorType {
    Sampler = 0,
    CombinedImageSampler = 1,
    SampledImage = 2,
    StorageImage = 3,
    UniformBuffer = 6,
    StorageBuffer = 7,
}

/// Shader stage bits, Vulkan's values, or-ed together.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum ShaderStage {
    Vertex = 1,
    Fragment = 16,
    Compute = 32,
}

/// One binding of a descriptor set layout.
#[hxn::data]
pub struct LayoutBinding {
    pub binding: i32,
    pub kind: DescriptorType,
    /// `ShaderStage` bits.
    pub stages: i32,
}

/// One descriptor set's bindings: one of hxnsl's bind groups.
#[hxn::data]
pub struct SetLayout {
    pub bindings: Vec<LayoutBinding>,
}

/// One vertex attribute: `location` is the shader's (hxnsl's `INPUT_<name>`).
#[hxn::data]
pub struct VertexAttribute {
    pub location: i32,
    pub binding: i32,
    pub format: Format,
    pub offset: i32,
}

/// One vertex buffer binding: its stride in bytes, and whether it advances per instance.
#[hxn::data]
pub struct VertexBinding {
    pub binding: i32,
    pub stride: i32,
    pub per_instance: bool,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum Topology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum LoadOp {
    Load = 0,
    Clear = 1,
    DontCare = 2,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum BindPoint {
    Graphics = 0,
    Compute = 1,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum Filter {
    Nearest = 0,
    Linear = 1,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum AddressMode {
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
}

/// A Vulkan instance: the loader, and the GPUs it finds.
#[hxn::interface]
pub struct Instance {
    shared: Arc<InstanceShared>,
    devices: Vec<vk::PhysicalDevice>,
}

#[hxn::methods]
impl Instance {
    /// An instance for the application `name`. Fails where no Vulkan loader or driver is
    /// installed. Asks for Vulkan 1.3, which the driver may lower.
    pub fn create(name: &str) -> Result<Instance, String> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader: {e}"))?;
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|e| failed("listing instance extensions", e))?;
        let has = |name: &CStr| {
            available
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let mut extensions: Vec<*const c_char> = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        // MoltenVK lists its devices only to an instance that asks for portability ones.
        if has(ash::khr::portability_enumeration::NAME) {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        if has(ash::khr::get_physical_device_properties2::NAME) {
            extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }
        let app_name = CString::new(name).map_err(|_| "the name contains a NUL")?;
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&extensions)
            .flags(flags);
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|e| failed("creating the instance", e))?;
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| failed("listing devices", e))?;
        Ok(Instance {
            shared: Arc::new(InstanceShared {
                _entry: entry,
                instance,
            }),
            devices,
        })
    }

    /// The name of each GPU, in the order `createDevice` takes them.
    pub fn adapter_names(&self) -> Vec<String> {
        self.devices
            .iter()
            .map(|&pd| {
                let properties = unsafe { self.shared.instance.get_physical_device_properties(pd) };
                properties
                    .device_name_as_c_str()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// A device on GPU `index`, with one queue that does graphics and compute. Fails if the
    /// GPU lacks dynamic rendering or synchronization2.
    pub fn create_device(&self, index: i32) -> Result<Device, String> {
        let pd = *usize::try_from(index)
            .ok()
            .and_then(|i| self.devices.get(i))
            .ok_or_else(|| format!("there is no GPU {index}"))?;
        let instance = &self.shared.instance;
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let family = families
            .iter()
            .position(|f| {
                f.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .ok_or("the GPU has no queue for graphics and compute")? as u32;
        let available = unsafe { instance.enumerate_device_extension_properties(pd) }
            .map_err(|e| failed("listing device extensions", e))?;
        let has = |name: &CStr| {
            available
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let mut extensions: Vec<*const c_char> = Vec::new();
        for needed in [
            ash::khr::synchronization2::NAME,
            ash::khr::dynamic_rendering::NAME,
        ] {
            if !has(needed) {
                return Err(format!("the GPU lacks {}", needed.to_string_lossy()));
            }
            extensions.push(needed.as_ptr());
        }
        // A portability implementation such as MoltenVK requires its subset to be enabled.
        if has(ash::khr::portability_subset::NAME) {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::default().synchronization2(true);
        let mut rendering =
            vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default().dynamic_rendering(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut sync2)
            .push_next(&mut rendering);
        let device = unsafe { instance.create_device(pd, &info, None) }
            .map_err(|e| failed("creating the device", e))?;
        let queue = unsafe { device.get_device_queue(family, 0) };
        let memory = unsafe { instance.get_physical_device_memory_properties(pd) };
        Ok(Device {
            shared: Arc::new(DeviceShared {
                instance: self.shared.clone(),
                sync2: ash::khr::synchronization2::Device::new(instance, &device),
                rendering: ash::khr::dynamic_rendering::Device::new(instance, &device),
                device,
                queue: Mutex::new(queue),
                family,
                memory,
            }),
        })
    }
}

/// A logical device and its queue; every other object comes from one.
#[hxn::interface]
pub struct Device {
    shared: Arc<DeviceShared>,
}

#[hxn::methods]
impl Device {
    /// A buffer of `size` bytes in memory the CPU can map, zeroed, with `usage` bits.
    pub fn create_buffer(&self, size: i64, usage: i32) -> Result<Buffer, String> {
        let size = usize_of(size, "a buffer's size")?.max(1);
        let d = &self.shared.device;
        let buffer = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(vk::BufferUsageFlags::from_raw(usage as u32))
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|e| failed("creating the buffer", e))?;
        let needs = unsafe { d.get_buffer_memory_requirements(buffer) };
        let kind = self.shared.memory_type(
            needs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let memory = unsafe {
            d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(kind),
                None,
            )
        }
        .map_err(|e| failed("allocating the buffer's memory", e))?;
        unsafe { d.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| failed("binding the buffer's memory", e))?;
        let mapped = unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
            .map_err(|e| failed("mapping the buffer", e))? as usize;
        unsafe { std::ptr::write_bytes(mapped as *mut u8, 0, size) };
        Ok(Buffer {
            shared: self.shared.clone(),
            buffer,
            memory,
            size,
            mapped,
        })
    }

    /// A buffer holding a copy of `bytes`.
    pub fn create_buffer_with(&self, bytes: &[u8], usage: i32) -> Result<Buffer, String> {
        let buffer = self.create_buffer(bytes.len() as i64, usage)?;
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// A 2D image in device memory, in the undefined layout, with one view of all of it.
    pub fn create_image(&self, descriptor: ImageDescriptor) -> Result<Image, String> {
        let width = u32_of(descriptor.width, "an image's width")?;
        let height = u32_of(descriptor.height, "an image's height")?;
        let format = descriptor.format.raw();
        let d = &self.shared.device;
        let image = unsafe {
            d.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::from_raw(descriptor.usage as u32))
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(|e| failed("creating the image", e))?;
        let needs = unsafe { d.get_image_memory_requirements(image) };
        let kind = self.shared.memory_type(
            needs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let memory = unsafe {
            d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(kind),
                None,
            )
        }
        .map_err(|e| failed("allocating the image's memory", e))?;
        unsafe { d.bind_image_memory(image, memory, 0) }
            .map_err(|e| failed("binding the image's memory", e))?;
        let aspect = aspect_of(descriptor.format);
        let view = unsafe {
            d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(whole(aspect)),
                None,
            )
        }
        .map_err(|e| failed("creating the image's view", e))?;
        Ok(Image {
            shared: self.shared.clone(),
            image,
            memory,
            view,
            width,
            height,
            aspect,
        })
    }

    pub fn create_sampler(
        &self,
        min: Filter,
        mag: Filter,
        address: AddressMode,
    ) -> Result<Sampler, String> {
        let filter = |f: Filter| vk::Filter::from_raw(f as i32);
        let address = vk::SamplerAddressMode::from_raw(address as i32);
        let sampler = unsafe {
            self.shared.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .min_filter(filter(min))
                    .mag_filter(filter(mag))
                    .address_mode_u(address)
                    .address_mode_v(address)
                    .address_mode_w(address)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }
        .map_err(|e| failed("creating the sampler", e))?;
        Ok(Sampler {
            shared: self.shared.clone(),
            sampler,
        })
    }

    /// A shader module from SPIR-V: hxnsl's `spirv()`.
    pub fn create_shader_module(&self, spirv: &[u8]) -> Result<ShaderModule, String> {
        if !spirv.len().is_multiple_of(4) {
            return Err(format!("{} bytes are not SPIR-V words", spirv.len()));
        }
        let words: Vec<u32> = spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let module = unsafe {
            self.shared
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|e| failed("creating the shader module", e))?;
        Ok(ShaderModule {
            shared: self.shared.clone(),
            module,
        })
    }

    /// The descriptor set layouts of a pipeline, set 0 first: one per hxnsl bind group.
    pub fn create_set_layouts(&self, sets: Vec<SetLayout>) -> Result<SetLayouts, String> {
        let mut made = SetLayouts {
            shared: self.shared.clone(),
            layouts: Vec::new(),
            bindings: Vec::new(),
        };
        for set in &sets {
            let raw = set
                .bindings
                .iter()
                .map(|b| {
                    Ok(vk::DescriptorSetLayoutBinding::default()
                        .binding(u32_of(b.binding, "a binding")?)
                        .descriptor_type(vk::DescriptorType::from_raw(b.kind as i32))
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::from_raw(b.stages as u32)))
                })
                .collect::<Result<Vec<_>, String>>()?;
            // Dropping `made` on failure destroys the layouts made so far.
            let layout = unsafe {
                self.shared.device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&raw),
                    None,
                )
            }
            .map_err(|e| failed("creating a descriptor set layout", e))?;
            made.layouts.push(layout);
            made.bindings.push(
                set.bindings
                    .iter()
                    .map(|b| (b.binding as u32, b.kind))
                    .collect(),
            );
        }
        Ok(made)
    }

    /// A pipeline layout of all of `sets`.
    pub fn create_pipeline_layout(&self, sets: &SetLayouts) -> Result<PipelineLayout, String> {
        let layout = unsafe {
            self.shared.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets.layouts),
                None,
            )
        }
        .map_err(|e| failed("creating the pipeline layout", e))?;
        Ok(PipelineLayout {
            shared: self.shared.clone(),
            layout,
        })
    }

    /// A descriptor set of set `index` of `sets`, from a pool of its own.
    pub fn create_descriptor_set(
        &self,
        sets: &SetLayouts,
        index: i32,
    ) -> Result<DescriptorSet, String> {
        let at = usize::try_from(index)
            .ok()
            .filter(|&i| i < sets.layouts.len())
            .ok_or_else(|| format!("there is no set {index}"))?;
        let bindings = &sets.bindings[at];
        let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for (_, kind) in bindings {
            let ty = vk::DescriptorType::from_raw(*kind as i32);
            match sizes.iter_mut().find(|s| s.ty == ty) {
                Some(size) => size.descriptor_count += 1,
                None => sizes.push(vk::DescriptorPoolSize {
                    ty,
                    descriptor_count: 1,
                }),
            }
        }
        let d = &self.shared.device;
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| failed("creating the descriptor pool", e))?;
        let layouts = [sets.layouts[at]];
        let allocated = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        };
        let set = match allocated {
            Ok(set) => set[0],
            Err(e) => {
                unsafe { d.destroy_descriptor_pool(pool, None) };
                return Err(failed("allocating the descriptor set", e));
            }
        };
        Ok(DescriptorSet {
            shared: self.shared.clone(),
            pool,
            set,
            bindings: bindings.clone(),
        })
    }

    /// A compute pipeline running `entry` of `module`: hxnsl's entry point names (`main`).
    pub fn create_compute_pipeline(
        &self,
        module: &ShaderModule,
        entry: &str,
        layout: &PipelineLayout,
    ) -> Result<Pipeline, String> {
        let name = CString::new(entry).map_err(|_| "the entry point contains a NUL")?;
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module.module)
                    .name(&name),
            )
            .layout(layout.layout);
        let pipeline = unsafe {
            self.shared
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, e)| failed("creating the compute pipeline", e))?[0];
        Ok(Pipeline {
            shared: self.shared.clone(),
            pipeline,
        })
    }

    /// A graphics pipeline for dynamic rendering into one color attachment of `format`, with a
    /// dynamic viewport and scissor, no blending and no culling.
    #[allow(clippy::too_many_arguments)]
    pub fn create_graphics_pipeline(
        &self,
        vertex: &ShaderModule,
        vertex_entry: &str,
        fragment: &ShaderModule,
        fragment_entry: &str,
        layout: &PipelineLayout,
        format: Format,
        topology: Topology,
        bindings: Vec<VertexBinding>,
        attributes: Vec<VertexAttribute>,
    ) -> Result<Pipeline, String> {
        let vertex_name =
            CString::new(vertex_entry).map_err(|_| "an entry point contains a NUL")?;
        let fragment_name =
            CString::new(fragment_entry).map_err(|_| "an entry point contains a NUL")?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex.module)
                .name(&vertex_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment.module)
                .name(&fragment_name),
        ];
        let vertex_bindings = bindings
            .iter()
            .map(|b| {
                Ok(vk::VertexInputBindingDescription {
                    binding: u32_of(b.binding, "a vertex binding")?,
                    stride: u32_of(b.stride, "a stride")?,
                    input_rate: if b.per_instance {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let vertex_attributes = attributes
            .iter()
            .map(|a| {
                Ok(vk::VertexInputAttributeDescription {
                    location: u32_of(a.location, "a location")?,
                    binding: u32_of(a.binding, "a vertex binding")?,
                    format: a.format.raw(),
                    offset: u32_of(a.offset, "an offset")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_bindings)
            .vertex_attribute_descriptions(&vertex_attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::from_raw(topology as i32));
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [format.raw()];
        let mut rendering =
            vk::PipelineRenderingCreateInfoKHR::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout.layout)
            .push_next(&mut rendering);
        let pipeline = unsafe {
            self.shared
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, e)| failed("creating the graphics pipeline", e))?[0];
        Ok(Pipeline {
            shared: self.shared.clone(),
            pipeline,
        })
    }

    /// A primary command buffer, from a pool of its own, ready to record.
    pub fn create_command_buffer(&self) -> Result<CommandBuffer, String> {
        let d = &self.shared.device;
        let pool = unsafe {
            d.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.shared.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| failed("creating the command pool", e))?;
        let buffer = unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| failed("allocating the command buffer", e))?[0];
        unsafe {
            d.begin_command_buffer(
                buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| failed("beginning the command buffer", e))?;
        Ok(CommandBuffer {
            recorded: Arc::new(Recorded {
                shared: self.shared.clone(),
                pool,
                buffer,
            }),
            ended: false,
        })
    }

    /// Submits `commands`, ending it first, and waits for the GPU to finish them.
    #[hxn(blocking)]
    pub fn submit_and_wait(&self, commands: &mut CommandBuffer) -> Result<(), String> {
        let (fence, _) = self.submit_fenced(commands)?;
        let d = &self.shared.device;
        let waited = unsafe { d.wait_for_fences(&[fence], true, u64::MAX) };
        unsafe { d.destroy_fence(fence, None) };
        waited.map_err(|e| failed("waiting for the GPU", e))
    }

    /// Submits `commands`, ending it first. The future settles when the GPU has finished them.
    pub fn submit(&self, commands: &mut CommandBuffer) -> Result<Future<()>, String> {
        let (fence, recorded) = self.submit_fenced(commands)?;
        // The command buffer stays alive until the fence signals, whatever Haxe does with it.
        Ok(hxn_future::spawn_blocking(move || {
            let d = &recorded.shared.device;
            let waited = unsafe { d.wait_for_fences(&[fence], true, u64::MAX) };
            unsafe { d.destroy_fence(fence, None) };
            waited.map_err(|e| failed("waiting for the GPU", e))
        }))
    }
}

impl Device {
    fn submit_fenced(
        &self,
        commands: &mut CommandBuffer,
    ) -> Result<(vk::Fence, Arc<Recorded>), String> {
        commands.end_recording()?;
        let d = &self.shared.device;
        let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| failed("creating a fence", e))?;
        let buffers = [commands.recorded.buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
        let queue = self
            .shared
            .queue
            .lock()
            .map_err(|_| "the queue's lock is poisoned")?;
        let submitted = unsafe { d.queue_submit(*queue, &submits, fence) };
        drop(queue);
        if let Err(e) = submitted {
            unsafe { d.destroy_fence(fence, None) };
            return Err(failed("submitting", e));
        }
        Ok((fence, commands.recorded.clone()))
    }
}

fn aspect_of(format: Format) -> vk::ImageAspectFlags {
    match format {
        Format::D32Float => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

fn whole(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Memory the CPU maps for as long as the buffer lives.
#[hxn::interface]
pub struct Buffer {
    shared: Arc<DeviceShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    // The mapped address, kept as an integer so the handle can cross threads.
    mapped: usize,
}

#[hxn::methods]
impl Buffer {
    pub fn size(&self) -> i64 {
        self.size as i64
    }

    /// Copies `bytes` into the buffer at `offset`.
    pub fn write(&self, offset: i64, bytes: &[u8]) -> Result<(), String> {
        let start = self.range(offset, bytes.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.mapped as *mut u8).add(start),
                bytes.len(),
            )
        };
        Ok(())
    }

    /// `length` bytes from `offset`.
    pub fn read(&self, offset: i64, length: i64) -> Result<Vec<u8>, String> {
        let length = usize_of(length, "a length")?;
        let start = self.range(offset, length)?;
        Ok(
            unsafe { std::slice::from_raw_parts((self.mapped as *const u8).add(start), length) }
                .to_vec(),
        )
    }
}

impl Buffer {
    fn range(&self, offset: i64, length: usize) -> Result<usize, String> {
        let start = usize_of(offset, "an offset")?;
        if start.checked_add(length).is_none_or(|end| end > self.size) {
            return Err(format!(
                "{length} bytes at {start} run past the buffer's {}",
                self.size
            ));
        }
        Ok(start)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let d = &self.shared.device;
        unsafe {
            d.unmap_memory(self.memory);
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.memory, None);
        }
    }
}

/// A 2D image in device memory, with one view of all of it.
#[hxn::interface]
pub struct Image {
    shared: Arc<DeviceShared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    aspect: vk::ImageAspectFlags,
}

#[hxn::methods]
impl Image {
    pub fn width(&self) -> i32 {
        self.width as i32
    }

    pub fn height(&self) -> i32 {
        self.height as i32
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let d = &self.shared.device;
        unsafe {
            d.destroy_image_view(self.view, None);
            d.destroy_image(self.image, None);
            d.free_memory(self.memory, None);
        }
    }
}

#[hxn::interface]
pub struct Sampler {
    shared: Arc<DeviceShared>,
    sampler: vk::Sampler,
}

#[hxn::methods]
impl Sampler {}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_sampler(self.sampler, None) };
    }
}

#[hxn::interface]
pub struct ShaderModule {
    shared: Arc<DeviceShared>,
    module: vk::ShaderModule,
}

#[hxn::methods]
impl ShaderModule {}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_shader_module(self.module, None) };
    }
}

/// The descriptor set layouts of a pipeline, set 0 first.
#[hxn::interface]
pub struct SetLayouts {
    shared: Arc<DeviceShared>,
    layouts: Vec<vk::DescriptorSetLayout>,
    bindings: Vec<Vec<(u32, DescriptorType)>>,
}

#[hxn::methods]
impl SetLayouts {
    /// How many sets there are.
    pub fn count(&self) -> i32 {
        self.layouts.len() as i32
    }
}

impl Drop for SetLayouts {
    fn drop(&mut self) {
        for layout in &self.layouts {
            unsafe {
                self.shared
                    .device
                    .destroy_descriptor_set_layout(*layout, None)
            };
        }
    }
}

#[hxn::interface]
pub struct PipelineLayout {
    shared: Arc<DeviceShared>,
    layout: vk::PipelineLayout,
}

#[hxn::methods]
impl PipelineLayout {}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.shared
                .device
                .destroy_pipeline_layout(self.layout, None)
        };
    }
}

#[hxn::interface]
pub struct Pipeline {
    shared: Arc<DeviceShared>,
    pipeline: vk::Pipeline,
}

#[hxn::methods]
impl Pipeline {}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_pipeline(self.pipeline, None) };
    }
}

/// A descriptor set, written binding by binding before a command buffer binds it.
#[hxn::interface]
pub struct DescriptorSet {
    shared: Arc<DeviceShared>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    bindings: Vec<(u32, DescriptorType)>,
}

#[hxn::methods]
impl DescriptorSet {
    /// Points `binding`, a uniform or storage buffer, at `size` bytes of `buffer` from `offset`.
    pub fn write_buffer(
        &mut self,
        binding: i32,
        buffer: &Buffer,
        offset: i64,
        size: i64,
    ) -> Result<(), String> {
        let (binding, kind) = self.binding(binding)?;
        if kind != DescriptorType::UniformBuffer && kind != DescriptorType::StorageBuffer {
            return Err(format!("binding {binding} is not a buffer"));
        }
        let infos = [vk::DescriptorBufferInfo {
            buffer: buffer.buffer,
            offset: usize_of(offset, "an offset")? as u64,
            range: usize_of(size, "a size")? as u64,
        }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::from_raw(kind as i32))
            .buffer_info(&infos);
        unsafe { self.shared.device.update_descriptor_sets(&[write], &[]) };
        Ok(())
    }

    /// Points `binding`, a sampled or storage image, at `image` in `layout`.
    pub fn write_image(
        &mut self,
        binding: i32,
        image: &Image,
        layout: Layout,
    ) -> Result<(), String> {
        let (binding, kind) = self.binding(binding)?;
        if kind != DescriptorType::SampledImage && kind != DescriptorType::StorageImage {
            return Err(format!("binding {binding} is not an image"));
        }
        let infos = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: image.view,
            image_layout: layout.raw(),
        }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::from_raw(kind as i32))
            .image_info(&infos);
        unsafe { self.shared.device.update_descriptor_sets(&[write], &[]) };
        Ok(())
    }

    /// Points `binding`, a sampler, at `sampler`.
    pub fn write_sampler(&mut self, binding: i32, sampler: &Sampler) -> Result<(), String> {
        let (binding, kind) = self.binding(binding)?;
        if kind != DescriptorType::Sampler {
            return Err(format!("binding {binding} is not a sampler"));
        }
        let infos = [vk::DescriptorImageInfo {
            sampler: sampler.sampler,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&infos);
        unsafe { self.shared.device.update_descriptor_sets(&[write], &[]) };
        Ok(())
    }
}

impl DescriptorSet {
    fn binding(&self, binding: i32) -> Result<(u32, DescriptorType), String> {
        let binding = u32_of(binding, "a binding")?;
        self.bindings
            .iter()
            .find(|(b, _)| *b == binding)
            .copied()
            .ok_or_else(|| format!("the set's layout has no binding {binding}"))
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// A recorded command buffer and its pool, kept until the GPU has finished it.
struct Recorded {
    shared: Arc<DeviceShared>,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
}

impl Drop for Recorded {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_command_pool(self.pool, None) };
    }
}

/// Records commands until a `Device.submit`. Not thread-safe: its methods take `&mut self`.
#[hxn::interface]
pub struct CommandBuffer {
    recorded: Arc<Recorded>,
    ended: bool,
}

#[hxn::methods]
impl CommandBuffer {
    pub fn bind_pipeline(&mut self, point: BindPoint, pipeline: &Pipeline) -> Result<(), String> {
        let cb = self.recording()?;
        unsafe {
            self.recorded
                .shared
                .device
                .cmd_bind_pipeline(cb, bind_point(point), pipeline.pipeline)
        };
        Ok(())
    }

    /// Binds `set` as set `index` of `layout`: hxnsl's bind group `index`.
    pub fn bind_descriptor_set(
        &mut self,
        point: BindPoint,
        layout: &PipelineLayout,
        index: i32,
        set: &DescriptorSet,
    ) -> Result<(), String> {
        let cb = self.recording()?;
        unsafe {
            self.recorded.shared.device.cmd_bind_descriptor_sets(
                cb,
                bind_point(point),
                layout.layout,
                u32_of(index, "a set index")?,
                &[set.set],
                &[],
            )
        };
        Ok(())
    }

    pub fn dispatch(&mut self, x: i32, y: i32, z: i32) -> Result<(), String> {
        let cb = self.recording()?;
        unsafe {
            self.recorded.shared.device.cmd_dispatch(
                cb,
                u32_of(x, "a group count")?,
                u32_of(y, "a group count")?,
                u32_of(z, "a group count")?,
            )
        };
        Ok(())
    }

    /// A barrier on all of `buffer`: work in `src_stages` doing `src_access` finishes, and its
    /// writes are visible, before work in `dst_stages` doing `dst_access` starts.
    pub fn buffer_barrier(
        &mut self,
        buffer: &Buffer,
        src_stages: i32,
        src_access: i32,
        dst_stages: i32,
        dst_access: i32,
    ) -> Result<(), String> {
        let cb = self.recording()?;
        let barriers = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(stages(src_stages))
            .src_access_mask(access(src_access))
            .dst_stage_mask(stages(dst_stages))
            .dst_access_mask(access(dst_access))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        let dependency = vk::DependencyInfoKHR::default().buffer_memory_barriers(&barriers);
        unsafe {
            self.recorded
                .shared
                .sync2
                .cmd_pipeline_barrier2(cb, &dependency)
        };
        Ok(())
    }

    /// A barrier on all of `image` that also moves it from layout `from` to `to`.
    #[allow(clippy::too_many_arguments)]
    pub fn image_barrier(
        &mut self,
        image: &Image,
        src_stages: i32,
        src_access: i32,
        from: Layout,
        dst_stages: i32,
        dst_access: i32,
        to: Layout,
    ) -> Result<(), String> {
        let cb = self.recording()?;
        let barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(stages(src_stages))
            .src_access_mask(access(src_access))
            .dst_stage_mask(stages(dst_stages))
            .dst_access_mask(access(dst_access))
            .old_layout(from.raw())
            .new_layout(to.raw())
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .subresource_range(whole(image.aspect))];
        let dependency = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
        unsafe {
            self.recorded
                .shared
                .sync2
                .cmd_pipeline_barrier2(cb, &dependency)
        };
        Ok(())
    }

    /// A barrier on all memory, for work that touches more than one resource.
    pub fn memory_barrier(
        &mut self,
        src_stages: i32,
        src_access: i32,
        dst_stages: i32,
        dst_access: i32,
    ) -> Result<(), String> {
        let cb = self.recording()?;
        let barriers = [vk::MemoryBarrier2::default()
            .src_stage_mask(stages(src_stages))
            .src_access_mask(access(src_access))
            .dst_stage_mask(stages(dst_stages))
            .dst_access_mask(access(dst_access))];
        let dependency = vk::DependencyInfoKHR::default().memory_barriers(&barriers);
        unsafe {
            self.recorded
                .shared
                .sync2
                .cmd_pipeline_barrier2(cb, &dependency)
        };
        Ok(())
    }

    /// Begins rendering into `target`, which must be in the color attachment layout, starting
    /// from it as it is, cleared to `red`, `green`, `blue`, `alpha`, or undefined. Sets the
    /// viewport and scissor to all of it.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_rendering(
        &mut self,
        target: &Image,
        load: LoadOp,
        red: f64,
        green: f64,
        blue: f64,
        alpha: f64,
    ) -> Result<(), String> {
        let cb = self.recording()?;
        let attachments = [vk::RenderingAttachmentInfoKHR::default()
            .image_view(target.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::from_raw(load as i32))
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [red as f32, green as f32, blue as f32, alpha as f32],
                },
            })];
        let extent = vk::Extent2D {
            width: target.width,
            height: target.height,
        };
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let info = vk::RenderingInfoKHR::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&attachments);
        let shared = &self.recorded.shared;
        unsafe {
            shared.rendering.cmd_begin_rendering(cb, &info);
            shared.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            shared.device.cmd_set_scissor(cb, 0, &[area]);
        }
        Ok(())
    }

    pub fn end_rendering(&mut self) -> Result<(), String> {
        let cb = self.recording()?;
        unsafe { self.recorded.shared.rendering.cmd_end_rendering(cb) };
        Ok(())
    }

    pub fn bind_vertex_buffer(
        &mut self,
        binding: i32,
        buffer: &Buffer,
        offset: i64,
    ) -> Result<(), String> {
        let cb = self.recording()?;
        unsafe {
            self.recorded.shared.device.cmd_bind_vertex_buffers(
                cb,
                u32_of(binding, "a vertex binding")?,
                &[buffer.buffer],
                &[usize_of(offset, "an offset")? as u64],
            )
        };
        Ok(())
    }

    /// Draws `count` vertices from `first`, `instances` times.
    pub fn draw(&mut self, count: i32, instances: i32, first: i32) -> Result<(), String> {
        let cb = self.recording()?;
        unsafe {
            self.recorded.shared.device.cmd_draw(
                cb,
                u32_of(count, "a vertex count")?,
                u32_of(instances, "an instance count")?,
                u32_of(first, "a first vertex")?,
                0,
            )
        };
        Ok(())
    }

    /// Copies all of `image`, which must be in the transfer source layout, into `buffer`, rows
    /// packed.
    pub fn copy_image_to_buffer(&mut self, image: &Image, buffer: &Buffer) -> Result<(), String> {
        let cb = self.recording()?;
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: image.aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: image.width,
                height: image.height,
                depth: 1,
            },
        };
        unsafe {
            self.recorded.shared.device.cmd_copy_image_to_buffer(
                cb,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer,
                &[region],
            )
        };
        Ok(())
    }
}

impl CommandBuffer {
    fn recording(&self) -> Result<vk::CommandBuffer, String> {
        if self.ended {
            return Err("the command buffer was submitted; record into a new one".into());
        }
        Ok(self.recorded.buffer)
    }

    fn end_recording(&mut self) -> Result<(), String> {
        let cb = self.recording()?;
        unsafe { self.recorded.shared.device.end_command_buffer(cb) }
            .map_err(|e| failed("ending the command buffer", e))?;
        self.ended = true;
        Ok(())
    }
}

fn bind_point(point: BindPoint) -> vk::PipelineBindPoint {
    match point {
        BindPoint::Graphics => vk::PipelineBindPoint::GRAPHICS,
        BindPoint::Compute => vk::PipelineBindPoint::COMPUTE,
    }
}
