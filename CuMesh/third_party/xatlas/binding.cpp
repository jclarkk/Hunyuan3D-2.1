#include <torch/extension.h>
#include "xatlas.h"


namespace xatlas {


void check_tensor(const torch::Tensor& tensor, const std::string& name, torch::ScalarType type) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == type, name, " has incorrect data type");
}


bool ProgressCallbackTrampoline(xatlas::ProgressCategory category, int progress, void* userData) {
    // userData stores pointer to py::function
    auto* func = static_cast<py::function*>(userData);
    
    py::gil_scoped_acquire gil;
    
    try {
        (*func)(xatlas::StringForEnum(category), progress);
        return true; // true: continue
    } catch (py::error_already_set& e) {
        return false; // false: stop
    }
}


class XAtlasWrapper {

private:
    xatlas::Atlas* m_atlas;

public:
    XAtlasWrapper() {
        m_atlas = xatlas::Create();
    }

    ~XAtlasWrapper() {
        xatlas::Destroy(m_atlas);
    }

    void AddMesh(
        const torch::Tensor& vertices,
        const torch::Tensor& faces,
        std::optional<const torch::Tensor> normals,
        std::optional<const torch::Tensor> uvs
    ) {
        check_tensor(vertices, "vertices", torch::kFloat32);
        check_tensor(faces, "faces", torch::kInt32);
        
        // 1. Construct mesh declaration
        xatlas::MeshDecl meshDecl;
        meshDecl.vertexCount = static_cast<uint32_t>(vertices.size(0));
        meshDecl.vertexPositionData = vertices.data_ptr<float>();
        meshDecl.vertexPositionStride = sizeof(float) * 3;
        meshDecl.indexCount = static_cast<uint32_t>(faces.size(0) * 3);
        meshDecl.indexData = faces.data_ptr<int32_t>();
        meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
        if (normals.has_value()) {
            check_tensor(*normals, "normals", torch::kFloat32);
            meshDecl.vertexNormalData = normals->data_ptr<float>();
            meshDecl.vertexNormalStride = sizeof(float) * 3;
        }
        if (uvs.has_value()) {
            check_tensor(*uvs, "uvs", torch::kFloat32);
            meshDecl.vertexUvData = uvs->data_ptr<float>();
            meshDecl.vertexUvStride = sizeof(float) * 2;
        }
        
        // 2. Add mesh to atlas
        xatlas::AddMeshError result = xatlas::AddMesh(m_atlas, meshDecl);
        if (result != xatlas::AddMeshError::Success) {
            throw std::runtime_error("Adding mesh failed: " + std::string(xatlas::StringForEnum(result)));
        }
    }

    void ComputeCharts(xatlas::ChartOptions options, std::optional<py::function> progressCallback) {
        if (progressCallback.has_value()) {
            xatlas::SetProgressCallback(m_atlas, &ProgressCallbackTrampoline, &(*progressCallback));
        }
        
        {
            py::gil_scoped_release gil;
            xatlas::ComputeCharts(m_atlas, options);
        }
        
        xatlas::SetProgressCallback(m_atlas, nullptr, nullptr);
    }

    void PackCharts(xatlas::PackOptions options, std::optional<py::function> progressCallback) {
        if (progressCallback.has_value()) {
            xatlas::SetProgressCallback(m_atlas, &ProgressCallbackTrampoline, &(*progressCallback));
        }

        {
            py::gil_scoped_release gil;
            xatlas::PackCharts(m_atlas, options);
        }

        xatlas::SetProgressCallback(m_atlas, nullptr, nullptr);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GetMesh(uint32_t index) {
        if (index >= m_atlas->meshCount) {
            throw std::out_of_range("Mesh index " + std::to_string(index) + " out of bounds for atlas with " + std::to_string(m_atlas->meshCount) + " meshes.");
        }

        auto const& mesh = m_atlas->meshes[index];

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

        auto mapping = torch::empty({(long)mesh.vertexCount}, options_int);
        auto faces = torch::empty({(long)mesh.indexCount / 3, 3}, options_int);
        auto uv = torch::empty({(long)mesh.vertexCount, 2}, options);

        int32_t* mappingPtr = mapping.data_ptr<int32_t>();
        int32_t* facesPtr = faces.data_ptr<int32_t>();
        float* uvPtr = uv.data_ptr<float>();

        float width = (float)m_atlas->width;
        float height = (float)m_atlas->height;

        for (uint32_t i = 0; i < mesh.vertexCount; ++i) {
            const auto& v = mesh.vertexArray[i];
            mappingPtr[i] = (int32_t)v.xref;
            
            if (width > 0 && height > 0) {
                uvPtr[i * 2 + 0] = v.uv[0] / width;
                uvPtr[i * 2 + 1] = v.uv[1] / height;
            } else {
                uvPtr[i * 2 + 0] = 0.0f;
                uvPtr[i * 2 + 1] = 0.0f;
            }
        }

        for (uint32_t i = 0; i < mesh.indexCount; ++i) {
            facesPtr[i] = (int32_t)mesh.indexArray[i];
        }

        return std::make_tuple(mapping, faces, uv);
    }
};


} // namespace xatlas


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "xatlas wrapper for PyTorch";

    py::class_<xatlas::ChartOptions>(m, "ChartOptions")
        .def(py::init<>())
        .def_readwrite("max_chart_area", &xatlas::ChartOptions::maxChartArea)
        .def_readwrite("max_boundary_length", &xatlas::ChartOptions::maxBoundaryLength)
        .def_readwrite("normal_deviation_weight", &xatlas::ChartOptions::normalDeviationWeight)
        .def_readwrite("roundness_weight", &xatlas::ChartOptions::roundnessWeight)
        .def_readwrite("straightness_weight", &xatlas::ChartOptions::straightnessWeight)
        .def_readwrite("normal_seam_weight", &xatlas::ChartOptions::normalSeamWeight)
        .def_readwrite("texture_seam_weight", &xatlas::ChartOptions::textureSeamWeight)
        .def_readwrite("max_cost", &xatlas::ChartOptions::maxCost)
        .def_readwrite("max_iterations", &xatlas::ChartOptions::maxIterations)
        .def_readwrite("use_input_mesh_uvs", &xatlas::ChartOptions::useInputMeshUvs)
        .def_readwrite("fix_winding", &xatlas::ChartOptions::fixWinding);

    py::class_<xatlas::PackOptions>(m, "PackOptions")
        .def(py::init<>())
        .def_readwrite("max_chart_size", &xatlas::PackOptions::maxChartSize)
        .def_readwrite("padding", &xatlas::PackOptions::padding)
        .def_readwrite("texels_per_unit", &xatlas::PackOptions::texelsPerUnit)
        .def_readwrite("resolution", &xatlas::PackOptions::resolution)
        .def_readwrite("bilinear", &xatlas::PackOptions::bilinear)
        .def_readwrite("block_align", &xatlas::PackOptions::blockAlign)
        .def_readwrite("brute_force", &xatlas::PackOptions::bruteForce)
        .def_readwrite("rotate_charts", &xatlas::PackOptions::rotateCharts)
        .def_readwrite("rotate_charts_to_axis", &xatlas::PackOptions::rotateChartsToAxis);

    py::class_<xatlas::XAtlasWrapper>(m, "Atlas")
        .def(py::init<>())
        .def("add_mesh", &xatlas::XAtlasWrapper::AddMesh)
        .def("compute_charts", &xatlas::XAtlasWrapper::ComputeCharts)
        .def("pack_charts", &xatlas::XAtlasWrapper::PackCharts)
        .def("get_mesh", &xatlas::XAtlasWrapper::GetMesh);
}
