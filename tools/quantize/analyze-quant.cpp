#include "common.h"
#include "llama.h"
#include "gguf.h"
#include "json-partial.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <fstream>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <regex>

// ftype options - limited to valid quantization types for size estimation
// These are the types that can actually be quantized via ggml_quantize_chunk
struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 4.78G, +0.4511 ppl @ Llama-3-8B",  },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 5.21G, +0.1316 ppl @ Llama-3-8B",  },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 5.65G, +0.1062 ppl @ Llama-3-8B",  },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 7.96G, +0.0026 ppl @ Llama-3-8B",  },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.96G, +3.5199 ppl @ Llama-3-8B",  },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 5.33G, +0.0569 ppl @ Llama-3-8B",  },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 6.14G, +0.0217 ppl @ Llama-3-8B",  },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "TQ1_0",    LLAMA_FTYPE_MOSTLY_TQ1_0,    " 1.69 bpw ternarization",           },
    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },
    { "MXFP4",    LLAMA_FTYPE_MOSTLY_MXFP4_MOE," MXFP4 quantization",               },
};

// Quantization type information
// Only includes types that can actually be quantized via ggml_quantize_chunk
struct QuantInfo {
    std::string name;
    ggml_type type;
    int block_size;
    int type_size;
};

static const QuantInfo QUANT_TYPES_ARRAY[] = {
    // Basic quantization types
    {"Q4_0",     GGML_TYPE_Q4_0,     32, 18},
    {"Q4_1",     GGML_TYPE_Q4_1,     32, 20},
    {"Q5_0",     GGML_TYPE_Q5_0,     32, 22},
    {"Q5_1",     GGML_TYPE_Q5_1,     32, 24},
    {"Q8_0",     GGML_TYPE_Q8_0,     32, 34},
    // K-quants
    {"Q2_K",     GGML_TYPE_Q2_K,    256, 84},
    {"Q3_K",     GGML_TYPE_Q3_K,    256, 110},
    {"Q4_K",     GGML_TYPE_Q4_K,    256, 144},
    {"Q5_K",     GGML_TYPE_Q5_K,    256, 176},
    {"Q6_K",     GGML_TYPE_Q6_K,    256, 210},
    // IQ types
    {"IQ2_XXS",  GGML_TYPE_IQ2_XXS, 256, 66},
    {"IQ2_XS",   GGML_TYPE_IQ2_XS,  256, 74},
    {"IQ3_XXS",  GGML_TYPE_IQ3_XXS, 256, 98},
    {"IQ1_S",    GGML_TYPE_IQ1_S,   256, 50},
    {"IQ4_NL",   GGML_TYPE_IQ4_NL,   32, 18},
    {"IQ3_S",    GGML_TYPE_IQ3_S,   256, 110},
    {"IQ2_S",    GGML_TYPE_IQ2_S,   256, 82},
    {"IQ4_XS",   GGML_TYPE_IQ4_XS,  256, 136},
    {"IQ1_M",    GGML_TYPE_IQ1_M,   256, 56},
    // Ternary quantization
    {"TQ1_0",    GGML_TYPE_TQ1_0,   256, 48},
    {"TQ2_0",    GGML_TYPE_TQ2_0,   256, 64},
    // MXFP4
    {"MXFP4",    GGML_TYPE_MXFP4,    32, 17},
};

static const std::vector<QuantInfo> QUANT_TYPES(QUANT_TYPES_ARRAY,
    QUANT_TYPES_ARRAY + sizeof(QUANT_TYPES_ARRAY) / sizeof(QUANT_TYPES_ARRAY[0]));

// Convert quantization type name string to ggml_type
static ggml_type quant_name_to_type(const std::string & name) {
    std::string upper_name = name;
    for (auto & c : upper_name) {
        c = std::toupper(c);
    }
    for (const auto & qi : QUANT_TYPES) {
        std::string qi_upper = qi.name;
        for (auto & c : qi_upper) {
            c = std::toupper(c);
        }
        if (qi_upper == upper_name) {
            return qi.type;
        }
    }
    return GGML_TYPE_COUNT;
}

// Convert ggml_type to string name
static const char * quant_type_to_name(ggml_type type) {
    for (const auto & qi : QUANT_TYPES) {
        if (qi.type == type) {
            return qi.name.c_str();
        }
    }
    return "UNKNOWN";
}

// Get block size for a quantization type
static int get_block_size(ggml_type type) {
    for (const auto & qi : QUANT_TYPES) {
        if (qi.type == type) {
            return qi.block_size;
        }
    }
    return 1;
}

// Get type size for a quantization type
static int get_type_size(ggml_type type) {
    for (const auto & qi : QUANT_TYPES) {
        if (qi.type == type) {
            return qi.type_size;
        }
    }
    return 4;
}

// Parse ftype string to llama_ftype (copied from quantize.cpp)
static bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (const auto & it : QUANT_OPTIONS) {
        if (ftype_str == it.name) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (const auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

// Convert llama_ftype to default ggml_type (based on llama-quant.cpp)
static ggml_type ftype_to_ggml_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: return GGML_TYPE_F32;
    }
}

// Parse ggml_type from string
static ggml_type parse_ggml_type(const std::string & type_str) {
    std::string upper_str = type_str;
    for (auto & c : upper_str) {
        c = std::toupper(c);
    }
    for (const auto & qi : QUANT_TYPES) {
        std::string qi_upper = qi.name;
        for (auto & c : qi_upper) {
            c = std::toupper(c);
        }
        if (qi_upper == upper_str) {
            return qi.type;
        }
    }
    return GGML_TYPE_COUNT;
}

// Represents a unique tensor shape
// For quantization size estimation, shapes with the same element count produce
// identical quantized sizes, so we deduplicate by element count, not exact dimensions.
struct TensorShape {
    std::vector<int64_t> dims;
    size_t elements;
    
    // For deduplication purposes, shapes are equal if they have the same element count
    // This is because quantization size only depends on element count, not shape
    bool operator==(const TensorShape& other) const {
        return elements == other.elements;
    }
    
    // Calculate the original size (assuming BF16 or F16)
    size_t original_size() const {
        return elements * 2;  // 2 bytes per element for BF16/F16
    }
    
    // Calculate quantized size for a given type (theoretical, for fallback)
    size_t quantized_size_theoretical(ggml_type type) const {
        int block_sz = get_block_size(type);
        int type_sz = get_type_size(type);
        size_t num_blocks = (elements + block_sz - 1) / block_sz;
        return num_blocks * type_sz;
    }
    
    // Get a hash key for the shape (for display purposes, shows actual dims)
    std::string to_string() const {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) ss << ",";
            ss << dims[i];
        }
        ss << "]";
        return ss.str();
    }
};

// Hash for TensorShape - hash by element count for deduplication
struct TensorShapeHash {
    size_t operator()(const TensorShape& s) const {
        return std::hash<size_t>{}(s.elements);
    }
};

// Tensor category for reporting
enum class TensorCategory {
    ATTENTION,
    FFN,
    OTHER
};

static TensorCategory classify_tensor(const std::string & name) {
    std::string lower_name = name;
    for (auto & c : lower_name) {
        c = std::tolower(c);
    }
    
    if (lower_name.find("attn") != std::string::npos) {
        return TensorCategory::ATTENTION;
    }
    if (lower_name.find("ffn") != std::string::npos ||
        lower_name.find("feed_forward") != std::string::npos ||
        lower_name.find("mlp") != std::string::npos) {
        return TensorCategory::FFN;
    }
    return TensorCategory::OTHER;
}

// Check if a tensor should be quantized (matching llama-quant.cpp logic)
// Returns true if the tensor should be quantized, false if it should stay at original type
static bool should_quantize_tensor(const std::string & name, int n_dims) {
    // Must end with "weight"
    if (name.size() < 6 || name.rfind("weight") != name.size() - 6) {
        return false;
    }
    
    // quantize only 2D and 3D tensors (experts)
    if (n_dims < 2) {
        return false;
    }
    
    // do not quantize norm tensors
    if (name.find("_norm.weight") != std::string::npos) {
        return false;
    }
    
    // do not quantize expert gating tensors
    if (name.find("ffn_gate_inp.weight") != std::string::npos) {
        return false;
    }
    
    // these are very small (e.g. 4x4)
    if (name.find("altup") != std::string::npos) {
        return false;
    }
    if (name.find("laurel") != std::string::npos) {
        return false;
    }
    
    // these are not too big so keep them as is
    if (name.find("per_layer_model_proj") != std::string::npos) {
        return false;
    }
    
    // do not quantize positional embeddings and token types (BERT)
    // Note: we check for common patterns, actual implementation uses LLM_TN
    if (name.find("position_embd") != std::string::npos) {
        return false;
    }
    if (name.find("token_types") != std::string::npos) {
        return false;
    }
    
    // do not quantize Mamba's small yet 2D weights
    if (name.find("ssm_conv1d.weight") != std::string::npos) {
        return false;
    }
    if (name.find("shortconv.conv.weight") != std::string::npos) {
        return false;
    }
    
    // do not quantize RWKV's small yet 2D weights
    if (name.find("time_mix_first.weight") != std::string::npos ||
        name.find("time_mix_w0.weight") != std::string::npos ||
        name.find("time_mix_w1.weight") != std::string::npos ||
        name.find("time_mix_w2.weight") != std::string::npos ||
        name.find("time_mix_v0.weight") != std::string::npos ||
        name.find("time_mix_v1.weight") != std::string::npos ||
        name.find("time_mix_v2.weight") != std::string::npos ||
        name.find("time_mix_a0.weight") != std::string::npos ||
        name.find("time_mix_a1.weight") != std::string::npos ||
        name.find("time_mix_a2.weight") != std::string::npos ||
        name.find("time_mix_g1.weight") != std::string::npos ||
        name.find("time_mix_g2.weight") != std::string::npos ||
        name.find("time_mix_decay_w1.weight") != std::string::npos ||
        name.find("time_mix_decay_w2.weight") != std::string::npos ||
        name.find("time_mix_lerp_fused.weight") != std::string::npos) {
        return false;
    }
    
    // do not quantize relative position bias (T5)
    if (name.find("attn_rel_b.weight") != std::string::npos) {
        return false;
    }
    
    // do not quantize specific multimodal tensors
    if (name.find(".position_embd.") != std::string::npos) {
        return false;
    }
    
    return true;
}

static const char * category_to_string(TensorCategory cat) {
    switch (cat) {
        case TensorCategory::ATTENTION: return "attention";
        case TensorCategory::FFN: return "ffn";
        case TensorCategory::OTHER: return "other";
    }
    return "unknown";
}

// Extract layer number from tensor name
static int extract_layer(const std::string & name) {
    std::regex pattern(R"(blk\.(\d+)\.)");
    std::smatch match;
    if (std::regex_search(name, match, pattern)) {
        return std::stoi(match[1]);
    }
    // Try other patterns
    std::regex pattern2(R"((?:layers?|model\.layers?)\.(\d+)\.)");
    if (std::regex_search(name, match, pattern2)) {
        return std::stoi(match[1]);
    }
    return -1;  // Non-layer tensor
}

// Tensor information from the model
struct TensorInfo {
    std::string name;
    TensorShape shape;
    std::string original_type;
    ggml_type original_ggml_type;  // Original ggml type for non-quantized tensors
    size_t original_size;
    TensorCategory category;
    int layer;
    int n_dims;  // Number of dimensions
    bool quantizable;  // Whether this tensor should be quantized
    size_t shape_idx;  // Index into unique_shapes array
};

// Measurement for a unique shape
struct ShapeMeasurement {
    TensorShape shape;
    std::unordered_map<std::string, size_t> quantized_sizes;  // quant_type_name -> size
    std::vector<float> sample_data;  // Sample F32 data for quantization
};

// Complete measurements for a model
struct ModelMeasurements {
    int version = 1;
    std::string model_name;
    std::string model_path;
    std::string architecture;
    size_t original_size = 0;
    std::vector<ShapeMeasurement> unique_shapes;
    std::vector<TensorInfo> tensors;
    
    std::string created_at;
    
    // Helper to find shape measurement index
    size_t find_shape_idx(const TensorShape & shape) {
        for (size_t i = 0; i < unique_shapes.size(); ++i) {
            if (unique_shapes[i].shape == shape) {
                return i;
            }
        }
        return unique_shapes.size();  // Not found
    }
};

// Quantization recipe with special tensor types
struct QuantizationRecipe {
    llama_ftype default_ftype;
    std::string default_type_name;
    ggml_type output_tensor_type;
    ggml_type token_embedding_type;
    std::vector<std::pair<std::string, std::string>> tensor_overrides;  // pattern -> type
    
    // Get the quantization type for a specific tensor (returns ggml_type)
    ggml_type get_ggml_type_for_tensor(const std::string & tensor_name, ggml_type default_ggml_type) const {
        // Check output.weight special case
        if (output_tensor_type < GGML_TYPE_COUNT && tensor_name == "output.weight") {
            return output_tensor_type;
        }
        
        // Check token_embd.weight special case
        if (token_embedding_type < GGML_TYPE_COUNT && tensor_name == "token_embd.weight") {
            return token_embedding_type;
        }
        
        // Check overrides using regex matching (like llama-quantize)
        // First match wins
        for (const auto & [pattern, type] : tensor_overrides) {
            try {
                std::regex re(pattern);
                if (std::regex_search(tensor_name, re)) {
                    return quant_name_to_type(type);
                }
            } catch (const std::regex_error &) {
                // If regex fails, fall back to substring match
                if (tensor_name.find(pattern) != std::string::npos) {
                    return quant_name_to_type(type);
                }
            }
        }
        return default_ggml_type;
    }
    
    // Get the quantization type name for a specific tensor (returns string)
    std::string get_type_name_for_tensor(const std::string & tensor_name, const std::string & default_type) const {
        // Check output.weight special case
        if (output_tensor_type < GGML_TYPE_COUNT && tensor_name == "output.weight") {
            return quant_type_to_name(output_tensor_type);
        }
        
        // Check token_embd.weight special case
        if (token_embedding_type < GGML_TYPE_COUNT && tensor_name == "token_embd.weight") {
            return quant_type_to_name(token_embedding_type);
        }
        
        // Check overrides using regex matching (like llama-quantize)
        // First match wins
        for (const auto & [pattern, type] : tensor_overrides) {
            try {
                std::regex re(pattern);
                if (std::regex_search(tensor_name, re)) {
                    return type;
                }
            } catch (const std::regex_error &) {
                // If regex fails, fall back to substring match
                if (tensor_name.find(pattern) != std::string::npos) {
                    return type;
                }
            }
        }
        return default_type;
    }
};

// Size estimation results
struct SizeEstimate {
    size_t total_original_size = 0;
    size_t total_quantized_size = 0;
    double compression_ratio = 0.0;
    
    std::unordered_map<std::string, size_t> size_per_quant_type;
    std::unordered_map<std::string, size_t> size_per_category;  // attention/ffn/other
    std::unordered_map<int, size_t> size_per_layer;
    
    // Per-tensor breakdown
    std::vector<std::tuple<std::string, size_t, size_t, std::string>> tensor_details;  // name, orig_size, quant_size, quant_type
};

// Format bytes to human readable string (binary: KiB, MiB, GiB, TiB)
static std::string format_bytes(size_t bytes) {
    const char * units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int unit_idx = 0;
    double size = (double)bytes;
    
    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return ss.str();
}

// Format bytes to human readable string (decimal: KB, MB, GB, TB)
static std::string format_bytes_decimal(size_t bytes) {
    const char * units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = (double)bytes;
    
    while (size >= 1000.0 && unit_idx < 4) {
        size /= 1000.0;
        unit_idx++;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return ss.str();
}

// Print usage information
static void print_usage(const char * executable) {
    printf("usage: %s [--help] [--out <file>] [--measurements <file>] [--tensor-type NAME=TYPE] [--tensor-type-file <file>] [--output-tensor-type TYPE] [--token-embedding-type TYPE] <model.gguf> [default_type]\n\n", executable);
    printf("  --help, -h:              Show this help message\n");
    printf("  --out <file>:            Generate LUT mode: save all quantization measurements to JSON file\n");
    printf("  --measurements <file>:   Estimate mode: load pre-computed measurements from JSON file\n");
    printf("  --tensor-type NAME=TYPE: Override quantization for matching tensors (llama-quantize compatible)\n");
    printf("                           Example: --tensor-type attn_q=Q8_0\n");
    printf("  --tensor-type-file <file>: File with tensor-type overrides (one per line)\n");
    printf("  --output-tensor-type TYPE: Quantization type for output.weight tensor\n");
    printf("  --token-embedding-type TYPE: Quantization type for token_embd.weight tensor\n");
    printf("\nPositional arguments:\n");
    printf("  <model.gguf>:            Input BF16 GGUF model file (required)\n");
    printf("  [default_type]:          Default quantization type (optional, defaults to Q8_0)\n");
    printf("\nModes:\n");
    printf("  1. Analysis mode (no --out or --measurements):\n");
    printf("     Quantizes only requested types, outputs size estimate. No file saved.\n\n");
    printf("  2. Generate LUT mode (--out):\n");
    printf("     Quantizes ALL possible types, saves measurements file, outputs estimate.\n\n");
    printf("  3. Estimate mode (--measurements):\n");
    printf("     Loads measurements file, outputs fast estimate without quantizing.\n");
    printf("\nAllowed quantization types:\n");
    for (const auto & qi : QUANT_OPTIONS) {
        if (qi.name != "COPY") {
            printf("  %-10s : %s\n", qi.name.c_str(), qi.desc.c_str());
        }
    }
    printf("\nExamples:\n");
    printf("  # Analysis mode - quick estimate\n");
    printf("  %s model.gguf --tensor-type attn_q=Q8_0 Q4_K_M\n\n", executable);
    printf("  # Generate LUT mode - save measurements for future use\n");
    printf("  %s model.gguf --out measurements.json --tensor-type attn_q=Q8_0 Q4_K_M\n\n", executable);
    printf("  # Estimate mode - fast lookup using saved measurements\n");
    printf("  %s model.gguf --measurements measurements.json --tensor-type attn_q=Q8_0 Q4_K_M\n", executable);
    exit(0);
}

// Parse tensor-type file
static bool parse_tensor_type_file(const std::string & filepath, std::vector<std::string> & tensor_types) {
    std::ifstream in(filepath);
    if (!in) {
        fprintf(stderr, "Error: failed to open tensor-type file: %s\n", filepath.c_str());
        return false;
    }
    
    std::string line;
    while (std::getline(in, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        tensor_types.push_back(line);
    }
    
    return true;
}

// Extract shape from GGML tensor
static TensorShape extract_tensor_shape(const ggml_tensor * tensor) {
    TensorShape shape;
    shape.elements = 1;
    
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 0) {
            shape.dims.push_back(tensor->ne[i]);
            shape.elements *= tensor->ne[i];
        } else {
            break;
        }
    }
    
    return shape;
}

// Load model and extract tensor information
static ModelMeasurements analyze_model(const std::string & model_path) {
    ModelMeasurements measurements;
    measurements.model_path = model_path;
    measurements.model_name = std::filesystem::path(model_path).filename().string();
    
    // Get current time
    time_t now = time(nullptr);
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    measurements.created_at = time_buf;
    
    printf("Loading model: %s\n", model_path.c_str());
    
    // Initialize GGML context
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "Error: failed to load GGUF model: %s\n", model_path.c_str());
        exit(1);
    }
    
    // Get architecture
    int arch_idx = gguf_find_key(ctx_gguf, "general.architecture");
    if (arch_idx >= 0) {
        measurements.architecture = gguf_get_val_str(ctx_gguf, arch_idx);
    }
    
    // Get model size
    int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    
    printf("Found %lld tensors\n", (long long)n_tensors);
    
    // Track unique shapes
    std::unordered_map<TensorShape, size_t, TensorShapeHash> shape_to_idx;
    
    // Iterate through all tensors
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
        
        TensorShape shape = extract_tensor_shape(tensor);
        
        // Get tensor type
        ggml_type tensor_type = tensor->type;
        const char * type_name = ggml_type_name(tensor_type);
        
        // Calculate original size
        size_t tensor_size = ggml_nbytes(tensor);
        measurements.original_size += tensor_size;
        
        // Check if shape already exists
        auto it = shape_to_idx.find(shape);
        size_t shape_idx;
        if (it != shape_to_idx.end()) {
            shape_idx = it->second;
        } else {
            shape_idx = measurements.unique_shapes.size();
            shape_to_idx[shape] = shape_idx;
            
            ShapeMeasurement sm;
            sm.shape = shape;
            
            // Create sample F32 data for quantization
            // Use a reasonable sample size - if tensor is huge, sample a subset
            size_t sample_size = std::min(shape.elements, (size_t)100000);
            sm.sample_data.resize(sample_size);
            
            // Generate pseudo-random but deterministic sample data
            for (size_t j = 0; j < sample_size; ++j) {
                // Use a simple hash-based pattern that varies across positions
                double val = std::sin((double)j * 0.1) * 0.5 + 0.5;  // 0.0 to 1.0
                sm.sample_data[j] = (float)val;
            }
            
            measurements.unique_shapes.push_back(sm);
            
            printf("Unique element count %zu: %s (%zu elements, %s)\n",
                   shape_idx, shape.to_string().c_str(),
                   shape.elements, format_bytes(tensor_size).c_str());
        }
        
        // Add tensor info
        TensorInfo ti;
        ti.name = name;
        ti.shape = shape;
        ti.original_type = type_name;
        ti.original_ggml_type = tensor_type;
        ti.original_size = tensor_size;
        ti.category = classify_tensor(name);
        ti.layer = extract_layer(name);
        ti.n_dims = (int)shape.dims.size();
        ti.quantizable = should_quantize_tensor(name, ti.n_dims);
        ti.shape_idx = shape_idx;
        
        measurements.tensors.push_back(ti);
    }
    
    printf("Total tensors: %zu, Unique element counts: %zu, Original size: %s\n",
           measurements.tensors.size(),
           measurements.unique_shapes.size(),
           format_bytes(measurements.original_size).c_str());
    
    // Clean up
    gguf_free(ctx_gguf);
    ggml_free(ctx);
    
    return measurements;
}

// Apply dimension compatibility fallback (based on llama-quant.cpp)
static ggml_type apply_dimension_fallback(ggml_type type, const TensorShape & shape) {
    int64_t nx = shape.dims.empty() ? 1 : shape.dims[0];
    int64_t qk_k = ggml_blck_size(type);
    
    if (nx % qk_k != 0) {
        switch (type) {
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:  type = GGML_TYPE_Q4_0; break;
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_IQ4_XS: type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q4_K:   type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:   type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:   type = GGML_TYPE_Q8_0;   break;
            default: break;
        }
        if (nx % ggml_blck_size(type) != 0) {
            type = GGML_TYPE_F16;
        }
    }
    return type;
}

// Quantize a sample tensor and measure size (actual quantization)
static size_t measure_quantized_size(const ShapeMeasurement & sm, ggml_type type, const std::string & /*tensor_name*/) {
    // Check if dimensions are compatible
    ggml_type actual_type = apply_dimension_fallback(type, sm.shape);
    
    if (actual_type != type) {
        const char * orig_name = quant_type_to_name(type);
        const char * fallback_name = quant_type_to_name(actual_type);
        if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
            int64_t nx = sm.shape.dims.empty() ? 1 : sm.shape.dims[0];
            int64_t qk_k = ggml_blck_size(type);
            printf("    Shape %s (dim0=%lld, blk=%lld): dimension fallback %s -> %s\n",
                   sm.shape.to_string().c_str(), (long long)nx, (long long)qk_k, orig_name, fallback_name);
        }
    }
    
    // For quantized types, perform actual quantization
    // Use the sample data we created earlier
    const float * f32_data = sm.sample_data.data();
    size_t sample_elements = sm.sample_data.size();
    
    // Ensure sample elements is sufficient for quantization
    int block_sz = ggml_blck_size(actual_type);
    if (block_sz == 0) block_sz = 1;
    if (sample_elements < (size_t)block_sz) {
        // Not enough data for even one block, use theoretical calculation
        return sm.shape.quantized_size_theoretical(actual_type);
    }
    
    // Calculate expected output size using ggml_row_size for accuracy
    size_t row_size = ggml_row_size(actual_type, sample_elements);
    
    // Allocate output buffer with some extra padding
    std::vector<uint8_t> output_buffer(row_size + 1024);
    
    // Perform actual quantization
    // Note: n_per_row is the number of elements per row (first dimension)
    int64_t n_per_row = sm.shape.dims.empty() ? sample_elements : sm.shape.dims[0];
    if (n_per_row == 0) n_per_row = 1;
    
    // If sample is smaller than full tensor, adjust n_per_row
    if (sample_elements < (size_t)n_per_row) {
        n_per_row = sample_elements;
    }
    
    // Ensure n_per_row is aligned to block size
    n_per_row = (n_per_row / block_sz) * block_sz;
    if (n_per_row == 0) n_per_row = block_sz;
    
    int64_t nrows = sample_elements / n_per_row;
    if (nrows == 0) nrows = 1;
    
    // Adjust sample_elements to be exactly nrows * n_per_row
    size_t actual_sample_elements = nrows * n_per_row;
    
    // Always provide an imatrix for types that require it (use ggml_quantize_requires_imatrix)
    // For size estimation purposes, we use a dummy imatrix (all 1.0s) since we only care
    // about the output size, not the quantization quality.
    std::vector<float> dummy_imatrix;
    const float * imatrix_ptr = nullptr;
    
    if (ggml_quantize_requires_imatrix(actual_type)) {
        dummy_imatrix.resize(actual_sample_elements, 1.0f);
        imatrix_ptr = dummy_imatrix.data();
    }
    
    // Quantize the sample data
    size_t actual_size = ggml_quantize_chunk(actual_type, f32_data, output_buffer.data(),
                                              0, nrows, n_per_row, imatrix_ptr);
    
    // Scale up to full tensor size
    size_t full_elements = sm.shape.elements;
    if (actual_sample_elements > 0) {
        double scale_factor = (double)full_elements / actual_sample_elements;
        return (size_t)(actual_size * scale_factor);
    }
    
    // Fallback to theoretical calculation
    return sm.shape.quantized_size_theoretical(actual_type);
}

// Perform quantization measurements for specific types
static void perform_measurements(ModelMeasurements & measurements, 
                                  const std::set<std::string> & requested_types) {
    printf("\nPerforming quantization measurements...\n");
    
    size_t total_measurements = measurements.unique_shapes.size() * requested_types.size();
    size_t completed = 0;
    
    for (auto & sm : measurements.unique_shapes) {
        for (const auto & type_name : requested_types) {
            ggml_type type = quant_name_to_type(type_name);
            if (type == GGML_TYPE_COUNT) {
                fprintf(stderr, "Warning: unknown quantization type: %s\n", type_name.c_str());
                continue;
            }
            
            // Measure quantized size using actual quantization
            std::string tensor_name = "shape_sample";  // Placeholder for fallback messages
            size_t quantized_size = measure_quantized_size(sm, type, tensor_name);
            sm.quantized_sizes[type_name] = quantized_size;
            
            completed++;
            if (completed % 10 == 0 || completed == total_measurements) {
                printf("Progress: %zu/%zu measurements (%.1f%%)\r", 
                       completed, total_measurements, 
                       100.0 * completed / total_measurements);
                fflush(stdout);
            }
        }
    }
    printf("\n");
}

// Calculate size estimate from measurements
static SizeEstimate calculate_estimate(const ModelMeasurements & measurements,
                                       const QuantizationRecipe & recipe) {
    SizeEstimate estimate;
    estimate.total_original_size = measurements.original_size;
    
    printf("\nCalculating size estimate...\n");
    
    // Get the default ggml_type from the ftype
    ggml_type default_ggml_type = ftype_to_ggml_type(recipe.default_ftype);
    
    for (const auto & tensor : measurements.tensors) {
        size_t quantized_size;
        std::string quant_type;
        
        // Check if this tensor should be quantized (matching llama-quant.cpp logic)
        if (!tensor.quantizable) {
            // Non-quantizable tensors stay at their original type and size
            quant_type = tensor.original_type;
            quantized_size = tensor.original_size;
        } else {
            // Get the quantization type for this tensor
            quant_type = recipe.get_type_name_for_tensor(tensor.name, recipe.default_type_name);
            ggml_type tensor_ggml_type = recipe.get_ggml_type_for_tensor(tensor.name, default_ggml_type);
            
            // Apply dimension compatibility fallback
            ggml_type actual_ggml_type = apply_dimension_fallback(tensor_ggml_type, tensor.shape);
            if (actual_ggml_type != tensor_ggml_type) {
                quant_type = quant_type_to_name(actual_ggml_type);
            }
            
            // Look up the quantized size from the shape measurement
            const ShapeMeasurement & sm = measurements.unique_shapes[tensor.shape_idx];
            auto it = sm.quantized_sizes.find(quant_type);
            
            if (it == sm.quantized_sizes.end()) {
                fprintf(stderr, "Warning: no measurement for type %s on tensor %s\n",
                        quant_type.c_str(), tensor.name.c_str());
                // Use original size as fallback
                quantized_size = tensor.original_size;
            } else {
                quantized_size = it->second;
            }
        }
        
        estimate.total_quantized_size += quantized_size;
        
        // Accumulate by quant type
        estimate.size_per_quant_type[quant_type] += quantized_size;
        
        // Accumulate by category
        const char * cat_name = category_to_string(tensor.category);
        estimate.size_per_category[cat_name] += quantized_size;
        
        // Accumulate by layer
        if (tensor.layer >= 0) {
            estimate.size_per_layer[tensor.layer] += quantized_size;
        }
        
        // Store tensor details
        estimate.tensor_details.push_back({tensor.name, tensor.original_size, quantized_size, quant_type});
    }
    
    estimate.compression_ratio = (double)estimate.total_original_size / estimate.total_quantized_size;
    
    return estimate;
}

// Save measurements to JSON file
static void save_measurements(const ModelMeasurements & measurements, const std::string & filepath) {
    printf("\nSaving measurements to: %s\n", filepath.c_str());
    
    nlohmann::json j;
    j["version"] = measurements.version;
    j["model"] = {
        {"name", measurements.model_name},
        {"path", measurements.model_path},
        {"architecture", measurements.architecture},
        {"original_size_bytes", measurements.original_size}
    };
    j["created_at"] = measurements.created_at;
    
    // Save unique shapes with their quantized sizes
    j["unique_shapes"] = nlohmann::json::array();
    for (const auto & sm : measurements.unique_shapes) {
        nlohmann::json shape_j;
        shape_j["shape"] = sm.shape.dims;
        shape_j["elements"] = sm.shape.elements;
        shape_j["original_size_bytes"] = sm.shape.original_size();
        shape_j["quantized_sizes"] = sm.quantized_sizes;
        j["unique_shapes"].push_back(shape_j);
    }
    
    // Save tensor list (references to unique shapes by index)
    j["tensors"] = nlohmann::json::array();
    for (const auto & ti : measurements.tensors) {
        nlohmann::json tensor_j;
        tensor_j["name"] = ti.name;
        tensor_j["shape"] = ti.shape.dims;
        tensor_j["shape_idx"] = ti.shape_idx;
        tensor_j["original_type"] = ti.original_type;
        tensor_j["original_size_bytes"] = ti.original_size;
        tensor_j["category"] = category_to_string(ti.category);
        tensor_j["layer"] = ti.layer;
        tensor_j["n_dims"] = ti.n_dims;
        tensor_j["quantizable"] = ti.quantizable;
        j["tensors"].push_back(tensor_j);
    }
    
    j["metadata"] = {
        {"tensor_count", measurements.tensors.size()},
        {"unique_shape_count", measurements.unique_shapes.size()}
    };
    
    std::ofstream out(filepath);
    if (!out) {
        fprintf(stderr, "Error: failed to open output file: %s\n", filepath.c_str());
        exit(1);
    }
    
    out << j.dump(2);
    printf("Saved %zu tensors (%zu unique shapes) to %s\n",
           measurements.tensors.size(),
           measurements.unique_shapes.size(),
           filepath.c_str());
}

// Load measurements from JSON file
static ModelMeasurements load_measurements(const std::string & filepath) {
    printf("Loading measurements from: %s\n", filepath.c_str());
    
    std::ifstream in(filepath);
    if (!in) {
        fprintf(stderr, "Error: failed to open measurements file: %s\n", filepath.c_str());
        exit(1);
    }
    
    nlohmann::json j;
    in >> j;
    
    ModelMeasurements measurements;
    
    if (j.contains("version")) {
        measurements.version = j["version"];
    }
    
    if (j.contains("model")) {
        measurements.model_name = j["model"]["name"];
        measurements.model_path = j["model"]["path"];
        measurements.architecture = j["model"]["architecture"];
        measurements.original_size = j["model"]["original_size_bytes"];
    }
    
    if (j.contains("created_at")) {
        measurements.created_at = j["created_at"];
    }
    
    // Load unique shapes
    if (j.contains("unique_shapes")) {
        for (const auto & shape_j : j["unique_shapes"]) {
            ShapeMeasurement sm;
            sm.shape.dims = shape_j["shape"].get<std::vector<int64_t>>();
            sm.shape.elements = shape_j["elements"];
            sm.quantized_sizes = shape_j["quantized_sizes"].get<std::unordered_map<std::string, size_t>>();
            measurements.unique_shapes.push_back(sm);
        }
    }
    
    // Load tensors
    if (j.contains("tensors")) {
        for (const auto & tensor_j : j["tensors"]) {
            TensorInfo ti;
            ti.name = tensor_j["name"];
            ti.shape.dims = tensor_j["shape"].get<std::vector<int64_t>>();
            ti.shape.elements = ti.shape.dims.empty() ? 0 :
                std::accumulate(ti.shape.dims.begin(), ti.shape.dims.end(), (int64_t)1, std::multiplies<int64_t>());
            ti.shape_idx = tensor_j["shape_idx"];
            ti.original_type = tensor_j["original_type"];
            ti.original_ggml_type = GGML_TYPE_F32;  // Default, will be overridden if present
            ti.original_size = tensor_j["original_size_bytes"];
            
            std::string cat = tensor_j["category"];
            if (cat == "attention") ti.category = TensorCategory::ATTENTION;
            else if (cat == "ffn") ti.category = TensorCategory::FFN;
            else ti.category = TensorCategory::OTHER;
            
            ti.layer = tensor_j["layer"];
            
            // Load new fields with defaults for backward compatibility
            ti.n_dims = tensor_j.contains("n_dims") ? tensor_j["n_dims"].get<int>() : (int)ti.shape.dims.size();
            ti.quantizable = tensor_j.contains("quantizable") ? tensor_j["quantizable"].get<bool>() :
                should_quantize_tensor(ti.name, ti.n_dims);
            
            measurements.tensors.push_back(ti);
        }
    }
    
    printf("Loaded %zu tensors (%zu unique shapes) from %s\n",
           measurements.tensors.size(),
           measurements.unique_shapes.size(),
           filepath.c_str());
    
    return measurements;
}

// Print size estimate results
static void print_estimate(const SizeEstimate & estimate, const QuantizationRecipe & recipe) {
    printf("\n");
    printf("================================================================================\n");
    printf("                    LLAMA-ANALYZE-QUANT ESTIMATION RESULTS\n");
    printf("================================================================================\n");
    
    printf("\nRecipe: %s default with %zu tensor-type overrides\n",
           recipe.default_type_name.c_str(), recipe.tensor_overrides.size());
    
    if (!recipe.tensor_overrides.empty()) {
        printf("\nTensor-type overrides:\n");
        for (const auto & [pattern, type] : recipe.tensor_overrides) {
            printf("  %-30s -> %s\n", pattern.c_str(), type.c_str());
        }
    }
    
    if (recipe.output_tensor_type < GGML_TYPE_COUNT) {
        printf("  output.weight          -> %s\n", quant_type_to_name(recipe.output_tensor_type));
    }
    if (recipe.token_embedding_type < GGML_TYPE_COUNT) {
        printf("  token_embd.weight      -> %s\n", quant_type_to_name(recipe.token_embedding_type));
    }
    
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("                              OVERALL ESTIMATES\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  Original Size:              %s / %s (%zu bytes)\n",
           format_bytes(estimate.total_original_size).c_str(),
           format_bytes_decimal(estimate.total_original_size).c_str(),
           estimate.total_original_size);
    printf("  Quantized Size:             %s / %s (%zu bytes)\n",
           format_bytes(estimate.total_quantized_size).c_str(),
           format_bytes_decimal(estimate.total_quantized_size).c_str(),
           estimate.total_quantized_size);
    printf("  Compression Ratio:          %.2fx\n", estimate.compression_ratio);
    
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("                        SIZE BY QUANTIZATION TYPE\n");
    printf("--------------------------------------------------------------------------------\n");
    
    // Sort by size descending
    std::vector<std::pair<std::string, size_t>> sorted_types(estimate.size_per_quant_type.begin(),
                                                              estimate.size_per_quant_type.end());
    std::sort(sorted_types.begin(), sorted_types.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    
    for (const auto & [type, size] : sorted_types) {
        double percent = 100.0 * size / estimate.total_quantized_size;
        printf("  %-10s: %14s / %14s - %5.1f%%\n",
               type.c_str(),
               format_bytes(size).c_str(),
               format_bytes_decimal(size).c_str(),
               percent);
    }
    
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("                          SIZE BY TENSOR CATEGORY\n");
    printf("--------------------------------------------------------------------------------\n");
    
    for (const auto & [category, size] : estimate.size_per_category) {
        double percent = 100.0 * size / estimate.total_quantized_size;
        printf("  %-12s: %14s / %14s - %5.1f%%\n",
               category.c_str(),
               format_bytes(size).c_str(),
               format_bytes_decimal(size).c_str(),
               percent);
    }
    
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("                            SIZE PER LAYER\n");
    printf("--------------------------------------------------------------------------------\n");
    
    std::vector<std::pair<int, size_t>> sorted_layers(estimate.size_per_layer.begin(),
                                                       estimate.size_per_layer.end());
    std::sort(sorted_layers.begin(), sorted_layers.end(),
              [](const auto & a, const auto & b) { return a.first < b.first; });
    
    int count = 0;
    for (const auto & [layer, size] : sorted_layers) {
        printf("  Layer %3d:   %14s / %14s\n",
               layer,
               format_bytes(size).c_str(),
               format_bytes_decimal(size).c_str());
        count++;
    }
    
    printf("\n");
    printf("================================================================================\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
    }
    
    llama_backend_init();
    
    // Parse arguments
    std::string input_path;
    std::string default_quant_type = "Q8_0";
    std::string measurements_path;
    std::string output_path;
    std::vector<std::string> tensor_type_args;
    ggml_type output_tensor_type = GGML_TYPE_COUNT;
    ggml_type token_embedding_type = GGML_TYPE_COUNT;
    
    int arg_idx = 1;
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "--help") == 0 || strcmp(argv[arg_idx], "-h") == 0) {
            print_usage(argv[0]);
        }
        else if (strcmp(argv[arg_idx], "--out") == 0) {
            if (++arg_idx >= argc) {
                fprintf(stderr, "Error: --out requires a filename argument\n");
                print_usage(argv[0]);
            }
            output_path = argv[arg_idx];
        }
        else if (strcmp(argv[arg_idx], "--measurements") == 0) {
            if (++arg_idx >= argc) {
                fprintf(stderr, "Error: --measurements requires a filename argument\n");
                print_usage(argv[0]);
            }
            measurements_path = argv[arg_idx];
        }
        else if (strcmp(argv[arg_idx], "--tensor-type") == 0) {
            if (++arg_idx >= argc) {
                fprintf(stderr, "Error: --tensor-type requires a NAME=TYPE argument\n");
                print_usage(argv[0]);
            }
            tensor_type_args.push_back(argv[arg_idx]);
        }
        else if (strcmp(argv[arg_idx], "--tensor-type-file") == 0) {
            if (++arg_idx >= argc) {
                fprintf(stderr, "Error: --tensor-type-file requires a filename argument\n");
                print_usage(argv[0]);
            }
            if (!parse_tensor_type_file(argv[arg_idx], tensor_type_args)) {
                exit(1);
            }
        }
        else if (strcmp(argv[arg_idx], "--output-tensor-type") == 0) {
            if (++arg_idx >= argc) {
                fprintf(stderr, "Error: --output-tensor-type requires a TYPE argument\n");
                print_usage(argv[0]);
            }
            output_tensor_type = parse_ggml_type(argv[arg_idx]);
            if (output_tensor_type == GGML_TYPE_COUNT) {
                fprintf(stderr, "Error: invalid quantization type: %s\n", argv[arg_idx]);
                print_usage(argv[0]);
            }
        }
        else if (strcmp(argv[arg_idx], "--token-embedding-type") == 0) {
            if (++arg_idx >= argc) {
                fprintf(stderr, "Error: --token-embedding-type requires a TYPE argument\n");
                print_usage(argv[0]);
            }
            token_embedding_type = parse_ggml_type(argv[arg_idx]);
            if (token_embedding_type == GGML_TYPE_COUNT) {
                fprintf(stderr, "Error: invalid quantization type: %s\n", argv[arg_idx]);
                print_usage(argv[0]);
            }
        }
        else if (argv[arg_idx][0] == '-') {
            fprintf(stderr, "Error: unknown option: %s\n", argv[arg_idx]);
            print_usage(argv[0]);
        }
        else {
            // Positional argument: model file
            input_path = argv[arg_idx];
            arg_idx++;
            
            // Optional positional: default quant type
            if (arg_idx < argc && argv[arg_idx][0] != '-') {
                std::string ftype_str;
                llama_ftype ftype;
                if (!try_parse_ftype(argv[arg_idx], ftype, ftype_str)) {
                    fprintf(stderr, "Error: invalid quantization type: %s\n", argv[arg_idx]);
                    exit(1);
                }
                default_quant_type = ftype_str;
                arg_idx++;
            }
            break;
        }
        arg_idx++;
    }
    
    if (input_path.empty()) {
        fprintf(stderr, "Error: No input model specified\n");
        print_usage(argv[0]);
    }
    
    // Validate mutual exclusivity
    if (!output_path.empty() && !measurements_path.empty()) {
        fprintf(stderr, "Warning: both --out and --measurements provided, using --measurements mode\n");
    }
    
    // Build quantization recipe
    QuantizationRecipe recipe;
    recipe.default_type_name = default_quant_type;
    recipe.output_tensor_type = output_tensor_type;
    recipe.token_embedding_type = token_embedding_type;
    
    // Parse default ftype
    llama_ftype ftype;
    if (!try_parse_ftype(default_quant_type, ftype, recipe.default_type_name)) {
        fprintf(stderr, "Error: invalid quantization type: %s\n", default_quant_type.c_str());
        exit(1);
    }
    recipe.default_ftype = ftype;
    
    for (const auto & arg : tensor_type_args) {
        size_t eq_pos = arg.find('=');
        if (eq_pos == std::string::npos || eq_pos == 0 || eq_pos == arg.length() - 1) {
            fprintf(stderr, "Error: invalid tensor-type format: %s (expected NAME=TYPE)\n", arg.c_str());
            exit(1);
        }
        std::string pattern = arg.substr(0, eq_pos);
        std::string type = arg.substr(eq_pos + 1);
        
        // Validate type name - check against QUANT_TYPES (ggml_type names)
        // Convert to uppercase for comparison
        std::string type_upper = type;
        for (auto & c : type_upper) {
            c = std::toupper(c);
        }
        
        ggml_type gtype = quant_name_to_type(type_upper);
        if (gtype == GGML_TYPE_COUNT) {
            // Also try via ftype for compatibility
            std::string ftype_str;
            llama_ftype temp_ftype;
            if (!try_parse_ftype(type, temp_ftype, ftype_str)) {
                fprintf(stderr, "Error: unknown quantization type: %s\n", type.c_str());
                exit(1);
            }
            type_upper = ftype_str;
        }
        
        recipe.tensor_overrides.push_back({pattern, type_upper});
    }
    
    // Determine mode
    if (!measurements_path.empty()) {
        // Estimate Mode: Load LUT and calculate estimate
        printf("================================================================================\n");
        printf("                              ESTIMATE MODE\n");
        printf("================================================================================\n");
        
        auto measurements = load_measurements(measurements_path);
        auto estimate = calculate_estimate(measurements, recipe);
        print_estimate(estimate, recipe);
    }
    else {
        // Analysis Mode or Generate LUT Mode
        printf("================================================================================\n");
        if (!output_path.empty()) {
            printf("                          GENERATE LUT MODE");
        } else {
            printf("                            ANALYSIS MODE");
        }
        printf("\n");
        printf("================================================================================\n");
        
        if (!output_path.empty()) {
            printf("Will perform all quantizations and save to: %s\n", output_path.c_str());
        } else {
            printf("Performing quantization on requested types only (no measurements file will be saved)\n");
        }
        
        // Build set of requested types
        std::set<std::string> requested_types;
        
        if (output_path.empty()) {
            // Analysis mode: only requested types
            for (const auto & [pattern, type] : recipe.tensor_overrides) {
                requested_types.insert(type);
            }
            requested_types.insert(recipe.default_type_name);
        } else {
            // Generate LUT mode: all types
            for (const auto & qi : QUANT_OPTIONS) {
                if (qi.name != "COPY") {
                    requested_types.insert(qi.name);
                }
            }
        }
        
        printf("Quantization types to measure: %zu\n", requested_types.size());
        for (const auto & type : requested_types) {
            printf("  - %s\n", type.c_str());
        }
        
        // Run analysis
        auto measurements = analyze_model(input_path);
        perform_measurements(measurements, requested_types);
        
        if (!output_path.empty()) {
            save_measurements(measurements, output_path);
        }
        
        // Always output the estimate for the requested recipe
        auto estimate = calculate_estimate(measurements, recipe);
        print_estimate(estimate, recipe);
    }
    
    llama_backend_free();
    
    return 0;
}
