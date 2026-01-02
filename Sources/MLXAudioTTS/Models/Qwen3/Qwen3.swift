//
//  Qwen3.swift
//  MLXAudio
//
//  Created by Prince Canuma on 29/12/2025.
//

import Foundation
@preconcurrency import MLX
import HuggingFace
import Tokenizers
import MLXLMCommon
import MLXFast
import MLXNN
import MLXAudioCodecs
import Combine


// MARK: - VyvoTTS special token IDs (Qwen3-based tokenizer)
let tokenizerLength = 151669
let startOfText = 151643
let endOfText = 151645
let startOfSpeech = tokenizerLength + 1  // 151670
let endOfSpeech = tokenizerLength + 2  // 151671
let startOfHuman = tokenizerLength + 3  // 151672
let endOfHuman = tokenizerLength + 4  // 151673
let startOfAI = tokenizerLength + 5  // 151674
let endOfAI = tokenizerLength + 6  // 151675
let padTokenId = tokenizerLength + 7  // 151676
let audioTokensStart = tokenizerLength + 10  // 151679

// MARK: - Error Types

public enum Qwen3Error: Error, LocalizedError {
    case modelNotInitialized(String)
    case generationFailed(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInitialized(let message):
            return "Model not initialized: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        }
    }
}

// MARK: - Generation Types

/// Information about the generation process.
public struct Qwen3GenerationInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let prefillTime: TimeInterval
    public let generateTime: TimeInterval
    public let tokensPerSecond: Double

    public var summary: String {
        """
        Prompt:     \(promptTokenCount) tokens, \(String(format: "%.2f", Double(promptTokenCount) / prefillTime)) tokens/s, \(String(format: "%.3f", prefillTime))s
        Generation: \(generationTokenCount) tokens, \(String(format: "%.2f", tokensPerSecond)) tokens/s, \(String(format: "%.3f", generateTime))s
        """
    }
}

/// Events emitted during audio generation.
public enum Qwen3Generation: Sendable {
    /// A generated token ID
    case token(Int)
    /// Generation statistics
    case info(Qwen3GenerationInfo)
    /// Final generated audio
    case audio(MLXArray)
}

// MARK: - Decode
func decodeAudioFromCodes(codeList: [Int], snacModel: SNAC) -> MLXArray {
    var layer1: [Int] = []
    var layer2: [Int] = []
    var layer3: [Int] = []

    let numGroups = (codeList.count + 1) / 7

    for i in 0..<numGroups {
        let baseIdx = 7 * i

        layer1.append(codeList[baseIdx])
        layer2.append(codeList[baseIdx + 1] - 4096)
        layer3.append(codeList[baseIdx + 2] - (2 * 4096))
        layer3.append(codeList[baseIdx + 3] - (3 * 4096))
        layer2.append(codeList[baseIdx + 4] - (4 * 4096))
        layer3.append(codeList[baseIdx + 5] - (5 * 4096))
        layer3.append(codeList[baseIdx + 6] - (6 * 4096))
    }

    let codes = [
        MLXArray(layer1).expandedDimensions(axis: 0),
        MLXArray(layer2).expandedDimensions(axis: 0),
        MLXArray(layer3).expandedDimensions(axis: 0)
    ]

    // SNAC decode returns [batch, channels, samples] - squeeze batch and channel dims
    let audioHat = snacModel.decode(codes).squeezed()
    return audioHat
}

func encodeAudioToCodes(audio: MLXArray, snacModel: SNAC) -> MLXArray {
    // Add batch and channel dimensions: [samples] -> [1, 1, samples]
    let audioExpanded = audio
        .expandedDimensions(axis: 0)
        .expandedDimensions(axis: 0)

    let codes = snacModel.encode(audioExpanded)

    let layer1 = codes[0].squeezed(axis: 0).asArray(Int.self)
    let layer2 = codes[1].squeezed(axis: 0).asArray(Int.self)
    let layer3 = codes[2].squeezed(axis: 0).asArray(Int.self)

    var codeList: [Int] = []
    let numGroups = layer1.count

    for i in 0..<numGroups {
        codeList.append(layer1[i])
        codeList.append(layer2[2 * i] + 4096)
        codeList.append(layer3[4 * i] + 2 * 4096)
        codeList.append(layer3[4 * i + 1] + 3 * 4096)
        codeList.append(layer2[2 * i + 1] + 4 * 4096)
        codeList.append(layer3[4 * i + 2] + 5 * 4096)
        codeList.append(layer3[4 * i + 3] + 6 * 4096)
    }

    return MLXArray(codeList).expandedDimensions(axis: 0)
}

// MARK: - Attention

public class Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
        let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a Float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta, scale: ropeScale
        )


    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)


        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            // Update cache and get full key/value history
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(
            output
        )
    }

}


// MARK: - MLP

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}


private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }


}


private class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers)
            .map { _ in TransformerBlock(args) }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}


public class Qwen3Model: Module, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var tokenizer: Tokenizer?
    public var _snacModel: SNAC?

    private let model: Qwen3ModelInner

    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // KVCacheDimensionProvider conformance
    public var numLayers: Int {
        return self.configuration.hiddenLayers
    }

    public init(_ args: Qwen3Configuration){
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0..<args.hiddenLayers).map {_ in args.kvHeads}
        self.model = Qwen3ModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func parseOutput(_ inputIds: MLXArray) -> [[Int]] {
        let tokenToRemove = endOfSpeech

        // First try to find START_OF_SPEECH
        var startIdx: Int? = nil

        // Create mask for START_OF_SPEECH
        let sosMask = inputIds .== startOfSpeech
        for i in 0..<sosMask.shape[0] {
            for j in 0..<sosMask.shape[1] {
                if sosMask[i, j].item(Int.self) != 0 {
                    startIdx = j
                }
            }
        }

        // If START_OF_SPEECH not found, look for START_OF_AI and find first audio token after it
        if startIdx == nil {
            let soaMask = inputIds .== startOfAI
            var soaIdx: Int? = nil
            for i in 0..<soaMask.shape[0] {
                for j in 0..<soaMask.shape[1] {
                    if soaMask[i, j].item(Int.self) != 0 {
                        soaIdx = j
                    }
                }
            }

            // Find first token >= audioTokensStart after START_OF_AI
            if let soaIdx = soaIdx {
                let rowList = inputIds[0].asArray(Int.self)
                for j in (soaIdx + 1)..<rowList.count {
                    if rowList[j] >= audioTokensStart {
                        startIdx = j - 1  // Start just before first audio token
                        break
                    }
                }
            }
        }

        var croppedTensor: MLXArray

        // Check if we found a starting point
        if let idx = startIdx {
            croppedTensor = inputIds[0..., (idx + 1)...]
        } else {
            croppedTensor = inputIds
        }

        // Process each row
        var processedRows: [MLXArray] = []

        for i in 0..<croppedTensor.shape[0] {
            let row = croppedTensor[i]
            let rowList = row.asArray(Int.self)

            // Filter out tokens to remove
            let maskedRow = rowList.filter { $0 != tokenToRemove }
            processedRows.append(MLXArray(maskedRow))
        }

        // Create code lists
        var codeLists: [[Int]] = []

        for row in processedRows {
            let rowLength = row.shape[0]
            let newLength = (rowLength / 7) * 7
            let trimmedRow = row[0..<newLength]

            // Subtract AUDIO_TOKENS_START from each token
            let trimmedList = trimmedRow.asArray(Int.self)
            let codeList = trimmedList.map { $0 - audioTokensStart }
            codeLists.append(codeList)
        }

        return codeLists
    }

    public func prepareInputIds(
        prompts: [String],
        voice: String? = nil,
        refAudio: MLXArray? = nil,
        refText: String? = nil
    ) -> (MLXArray, MLXArray) {

        var audioInputIds: MLXArray?
        var audioTranscriptIds: MLXArray?

        // Handle reference audio and text
        if let refAudio = refAudio, let refText = refText {
            print("\u{001B}[93mWARNING: Audio cloning doesn't work reliably on this model.\u{001B}[0m")

            guard let snacModel = self._snacModel else {
                fatalError("SNAC model not loaded. Call post_load_hook first.")
            }

            let codes = encodeAudioToCodes(audio: refAudio, snacModel: snacModel)
            audioInputIds = codes + audioTokensStart
            let encodedIds = tokenizer!.encode(text: refText)
            audioTranscriptIds = MLXArray(encodedIds.map { Int32($0) }).expandedDimensions(axis: 0)
        }

        // Apply voice prefix if provided
        var modifiedPrompts = prompts
        if let voice = voice {
            modifiedPrompts = prompts.map { "\(voice): \($0)" }
        }

        // Define special tokens
        let startToken = MLXArray([Int32(startOfHuman)]).expandedDimensions(axis: 0)
        let endTokens = MLXArray([Int32(endOfText), Int32(endOfHuman)]).expandedDimensions(axis: 0)

        // Encode all prompts
        var promptInputIds: [MLXArray] = []
        for prompt in modifiedPrompts {
            let encodedIds = tokenizer!.encode(text: prompt)
            let encoded = MLXArray(encodedIds.map { Int32($0) }).expandedDimensions(axis: 0)
            promptInputIds.append(encoded)
        }

        // Prepare batch with padding
        var batchInputIds: [MLXArray] = []
        let padToken = MLXArray([Int32(padTokenId)])
        let maxLen = promptInputIds.map { $0.shape[1] }.max() ?? 0

        for inputIds in promptInputIds {
            var modifiedInputIds: [MLXArray] = []

            // Add padding if needed
            let paddingLen = maxLen - inputIds.shape[1]
            if paddingLen > 0 {
                let padding = repeated(padToken, count: paddingLen, axis: 0)
                    .expandedDimensions(axis: 0)
                modifiedInputIds.append(padding)
            }

            // Add reference audio and transcript if provided
            if let audioInputIds = audioInputIds, let audioTranscriptIds = audioTranscriptIds {
                let audioStartTokens = MLXArray([
                    Int32(startOfAI), Int32(startOfSpeech)
                ]).expandedDimensions(axis: 0)

                let audioEndTokens = MLXArray([
                    Int32(endOfSpeech), Int32(endOfAI)
                ]).expandedDimensions(axis: 0)

                let refInputIds = concatenated([
                    startToken,
                    audioTranscriptIds,
                    endTokens,
                    audioStartTokens,
                    audioInputIds,
                    audioEndTokens
                ], axis: 1)

                modifiedInputIds.append(refInputIds)
            }

            // Add prompt with start/end tokens
            let onePromptInputIds = concatenated([
                startToken,
                inputIds,
                endTokens
            ], axis: 1)

            modifiedInputIds.append(onePromptInputIds)

            // Concatenate all parts for this prompt
            let fullInputIds = concatenated(modifiedInputIds, axis: 1)
            batchInputIds.append(fullInputIds)
        }

        // Concatenate all prompts in batch
        let finalBatchInputIds = concatenated(batchInputIds, axis: 0)

        // Create attention mask (False for pad tokens, True otherwise)
        let batchMask = finalBatchInputIds .!= padToken

        return (finalBatchInputIds, batchMask)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)

        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out
    }

    public var sampleRate: Int {
        return self.configuration.sampleRate
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    public func post_load_hook(model: Qwen3Model, modelDir: URL) async throws {
        if model.tokenizer == nil {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        }
        if model._snacModel == nil {
            model._snacModel = try await SNAC.fromPretrained("mlx-community/snac_24khz")
        }
    }

    // MARK: - Generation using MLXLMCommon evaluate pattern

    /// Generate audio from text using MLXLMCommon's evaluate-style token generation.
    ///
    /// This follows the pattern from MLXLMCommon's `generate` function with:
    /// - Configurable sampling (temperature, top-p)
    /// - Repetition penalty via LogitProcessor
    /// - KV cache for efficient generation
    ///
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - voice: Optional voice identifier (e.g., "en-us-1")
    ///   - parameters: Generation parameters (temperature, topP, maxTokens, etc.)
    /// - Returns: Generated audio as MLXArray
    public func generate(
        text: String,
        voice: String? = nil,
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 1200,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )
    ) async throws -> MLXArray {
        guard let snacModel = _snacModel else {
            throw Qwen3Error.modelNotInitialized("SNAC model not loaded")
        }
        guard tokenizer != nil else {
            throw Qwen3Error.modelNotInitialized("Tokenizer not loaded")
        }

        // Prepare input
        let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")

        let (inputIds, _) = prepareInputIds(prompts: [prompt], voice: voice)

        // Create sampler and processor from parameters
        let sampler = parameters.sampler()
        var processor = parameters.processor()

        // Initialize prompt tokens for processor
        let promptTokens = inputIds.squeezed(axis: 0)
        processor?.prompt(promptTokens)

        // Create KV cache
        let cache: [KVCache] = (0..<configuration.hiddenLayers).map { _ in
            KVCacheSimple()
        }

        // Track generated tokens
        var allTokens = inputIds
        let maxTokens = parameters.maxTokens ?? 1200

        // Prefill: process the prompt
        var logits = self(inputIds, cache: cache)

        // Generate tokens
        for i in 0..<maxTokens {
            // Get logits for the last position
            var lastLogits = logits[0..., -1, 0...]

            // Apply logit processor (repetition penalty)
            lastLogits = processor?.process(logits: lastLogits) ?? lastLogits

            // Sample next token
            let nextToken = sampler.sample(logits: lastLogits)

            // Notify processor of sampled token
            processor?.didSample(token: nextToken)

            // Check for end of speech
            let tokenValue = nextToken.item(Int.self)
            if tokenValue == endOfSpeech {
                break
            }

            // Append token to sequence
            let nextTokenExpanded = nextToken.reshaped([1, 1])
            allTokens = concatenated([allTokens, nextTokenExpanded], axis: 1)

            // Forward pass with just the new token (using cache)
            logits = self(nextTokenExpanded, cache: cache)

            // Periodically clear GPU cache
            if i % 50 == 0 {
                Memory.clearCache()
            }

            // Async evaluation for pipelining
            eval(logits)
        }

        // Parse output to audio codes
        let codeLists = parseOutput(allTokens)

        guard let codeList = codeLists.first, !codeList.isEmpty else {
            throw Qwen3Error.generationFailed("No audio codes generated")
        }

        // Decode audio using SNAC
        let audio = decodeAudioFromCodes(codeList: codeList, snacModel: snacModel)

        return audio
    }

    /// Generate audio with streaming token output.
    ///
    /// Returns an AsyncThrowingStream that yields generation events including tokens and final audio.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - voice: Optional voice identifier
    ///   - parameters: Generation parameters
    /// - Returns: AsyncThrowingStream of Qwen3Generation events
    public func generateStream(
        text: String,
        voice: String? = nil,
        parameters: GenerateParameters = GenerateParameters(
            maxTokens: 1200,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )
    ) -> AsyncThrowingStream<Qwen3Generation, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let snacModel = self._snacModel else {
                        throw Qwen3Error.modelNotInitialized("SNAC model not loaded")
                    }
                    guard self.tokenizer != nil else {
                        throw Qwen3Error.modelNotInitialized("Tokenizer not loaded")
                    }

                    let prompt = text.replacingOccurrences(of: "\\n", with: "\n")
                        .replacingOccurrences(of: "\\t", with: "\t")

                    let (inputIds, _) = self.prepareInputIds(prompts: [prompt], voice: voice)

                    let sampler = parameters.sampler()
                    var processor = parameters.processor()

                    let promptTokens = inputIds.squeezed(axis: 0)
                    processor?.prompt(promptTokens)

                    let cache: [KVCache] = (0..<self.configuration.hiddenLayers).map { _ in
                        KVCacheSimple()
                    }

                    var allTokens = inputIds
                    let maxTokens = parameters.maxTokens ?? 1200

                    let startTime = Date()

                    // Prefill
                    var logits = self(inputIds, cache: cache)
                    let prefillTime = Date().timeIntervalSince(startTime)

                    var tokenCount = 0
                    let generateStartTime = Date()

                    // Generate tokens
                    for _ in 0..<maxTokens {
                        if Task.isCancelled { break }

                        var lastLogits = logits[0..., -1, 0...]
                        lastLogits = processor?.process(logits: lastLogits) ?? lastLogits

                        let nextToken = sampler.sample(logits: lastLogits)
                        processor?.didSample(token: nextToken)

                        let tokenValue = nextToken.item(Int.self)
                        tokenCount += 1

                        // Yield token event
                        continuation.yield(.token(tokenValue))

                        if tokenValue == endOfSpeech {
                            break
                        }

                        let nextTokenExpanded = nextToken.reshaped([1, 1])
                        allTokens = concatenated([allTokens, nextTokenExpanded], axis: 1)

                        logits = self(nextTokenExpanded, cache: cache)

                        if tokenCount % 50 == 0 {
                            Memory.clearCache()
                        }

                        eval(logits)
                    }

                    let generateTime = Date().timeIntervalSince(generateStartTime)

                    // Parse and decode audio
                    let codeLists = self.parseOutput(allTokens)

                    guard let codeList = codeLists.first, !codeList.isEmpty else {
                        throw Qwen3Error.generationFailed("No audio codes generated")
                    }

                    let audio = decodeAudioFromCodes(codeList: codeList, snacModel: snacModel)
                    audio.eval()

                    // Yield completion info
                    let info = Qwen3GenerationInfo(
                        promptTokenCount: inputIds.shape[1],
                        generationTokenCount: tokenCount,
                        prefillTime: prefillTime,
                        generateTime: generateTime,
                        tokensPerSecond: Double(tokenCount) / generateTime
                    )
                    continuation.yield(.info(info))

                    // Yield final audio
                    continuation.yield(.audio(audio))

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public static func fromPretrained(_ modelRepo: String) async throws -> Qwen3Model {
        let client = HubClient.default
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "Qwen3Model", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }

        // Check if model is already fully cached (has weight files)
        let modelDir = try await resolveOrDownloadModel(
            client: client,
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors"
        )


        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: configData)

        let perLayerQuantization = config.perLayerQuantization

        let model = Qwen3Model(config)


        // Load weights from safetensors
        let weights = try loadWeights(from: modelDir)

        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize if needed

        if perLayerQuantization != nil {
            print("Applying quantizaiton from config...")

            if let perLayerQuant = perLayerQuantization {
                print(" Per-layer: \(perLayerQuant)")
            }

            quantize(model: model) { path, module in
                // Only quantize if scales exist for this layer
                if weights["\(path).scales"] != nil {
                    return perLayerQuantization?.quantization(layer: path)?.asTuple
                } else {
                    return nil
                }
            }
        }



        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])
        eval(model)

        try await model.post_load_hook(model: model, modelDir: modelDir)

        return model
    }
}

func loadWeights(from directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}

/// Resolves a model from cache or downloads it if not cached.
/// - Parameters:
///   - client: The HuggingFace Hub client
///   - cache: The HuggingFace cache
///   - repoID: The repository ID
///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
/// - Returns: The model directory URL
func resolveOrDownloadModel(
    client: HubClient,
    cache: HubCache,
    repoID: Repo.ID,
    requiredExtension: String
) async throws -> URL {
    // Use a persistent cache directory based on repo ID
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)

    // Check if model already exists with required files
    if FileManager.default.fileExists(atPath: modelDir.path) {
        let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let hasRequiredFiles = files?.contains { $0.pathExtension == requiredExtension } ?? false

        if hasRequiredFiles {
            print("Using cached model at: \(modelDir.path)")
            return modelDir
        }
    }

    // Create directory if needed
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

    print("Downloading model \(repoID)...")
    _ = try await client.downloadSnapshot(
        of: repoID,
        kind: .model,
        to: modelDir,
        revision: "main",
        progressHandler: { progress in
            print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
        }
    )

    print("Model downloaded to: \(modelDir.path)")
    return modelDir
}
