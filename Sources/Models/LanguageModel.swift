//
//  LanguageModel.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

import CoreML
import Generation
import Tokenizers
import Configurations

@available(macOS 15.0, iOS 18.0, *)
public class LanguageModel {
    public let model: MLModel

    public let minContextLength: Int
    public let maxContextLength: Int

    private var configuration: Configurations?
    private var _tokenizer: Tokenizer?

  

    public required init(
        model: MLModel,
        modelConfigURL: URL,
        tokenizerDataURL: URL,
        tokenizerConfigURL: URL? = nil,
        chatTemplateConfigURL: URL? = nil
    ) {
        self.model = model
        (minContextLength, maxContextLength) = Self.contextRange(from: model)
        
        do {
            configuration = try Configurations.loadConfigurations(modelConfigURL: modelConfigURL, tokenizerDataURL: tokenizerDataURL, tokenizerConfigURL: tokenizerConfigURL, chatTemplateConfigURL: chatTemplateConfigURL)
        } catch {
            fatalError("Erreur lors du chargement des configurations: \(error.localizedDescription)")
        }
    }

    public func resetState() async { }

    public func predictNextTokenScores(
        _ tokens: MLTensor,
        config: GenerationConfig
    ) async -> MLTensor {
        assert(tokens.rank == 2) // [batch, current sequence length]
        let tokenCount = tokens.shape[1]
        let padLength = maxContextLength - tokenCount
        let padding = MLTensor(repeating: Int32(config.padTokenId ?? 0), shape: [1, padLength])
        let inputIDs = MLTensor(concatenating: [tokens, padding], alongAxis: -1)
        var inputDictionary = [inputIdsName: inputIDs]
        if isRequiringAttentionMask {
            let mask = [Int32](repeating: 1, count: tokenCount) + [Int32](repeating: 0, count: padLength)
            let attentionMask = MLTensor(shape: inputIDs.shape, scalars: mask)
            inputDictionary[Keys.attentionMask] = attentionMask
        }
        let outputs = try! await model.prediction(from: inputDictionary)

        assert(outputs.keys.contains(Keys.logits))

        let scores = outputs[Keys.logits]!
        assert(scores.rank == 3)
        let tokenIndex = tokenCount - 1
        let nextTokenScores = scores[nil, tokenIndex, nil].expandingShape(at: 0)
        assert(nextTokenScores.rank == 3)
        assert(nextTokenScores.shape[0] == 1 && nextTokenScores.shape[1] == 1)
        return nextTokenScores
    }
}

@available(macOS 15.0, iOS 18.0, *)
private extension LanguageModel {
    static func contextRange(from model: MLModel) -> (min: Int, max: Int) {
        contextRange(from: model, inputKey: Keys.inputIds)
    }

    static func contextRange(from model: MLModel, inputKey: String) -> (min: Int, max: Int) {
        let inputDescription = model.modelDescription.inputDescriptionsByName[inputKey]
        
        print("Model Description : \(model.modelDescription)")
        
        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            fatalError("Cannot obtain shape information")   // Il est possible que les clefs dans Keys doivent être au format snake_case et non camelCase. Pour le savoir, se baser sur le retour du print juste au dessus
        }

        var minContextLength = 128
        var maxContextLength = 128

        switch shapeConstraint.type {
        case .enumerated:
            minContextLength = shapeConstraint.enumeratedShapes[0][1].intValue
            maxContextLength = minContextLength
        case .range:
            if let sizeRangeForDimension = inputDescription?.multiArrayConstraint?.shapeConstraint.sizeRangeForDimension {
                let lastAxis = sizeRangeForDimension.count - 1
                let range = sizeRangeForDimension[lastAxis] as? NSRange
                minContextLength = range?.location ?? 1
                maxContextLength = range?.length ?? 128
            }
        case .unspecified:
            break
        @unknown default:
            break
        }

        return (minContextLength, maxContextLength)
    }
}


@available(macOS 15.0, iOS 18.0, *)
extension LanguageModel {
    enum Keys {
        // Input keys
        static let inputIds = "inputIds"
        static let attentionMask = "attentionMask"
        static let causalMask = "causalMask"
        static let keyCache = "keyCache"
        static let valueCache = "valueCache"
        // Output keys
        static let logits = "logits"
        static let presentKeys = "presentKeys"
        static let presentValues = "presentValues"
    }
}

@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModel {
    static func loadCompiled(
        modelURL: URL,
        modelConfigURL: URL,
        tokenizerDataURL: URL,
        tokenizerConfigURL: URL? = nil,
        chatTemplateConfigURL: URL? = nil,
        computeUnits: MLComputeUnits = .all
    ) throws -> LanguageModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try MLModel(contentsOf: modelURL, configuration: config)    // Possible erreur "hit program assert". Supprimer l'application et ses fichiers de l'ordinateur de test et recompiler
        switch kvCacheAvailability(for: model) {
        case .statefulKVCache:
            print("KVCache")
            return LanguageModelWithStatefulKVCache(model: model, modelConfigURL: modelConfigURL, tokenizerDataURL: tokenizerDataURL, tokenizerConfigURL: tokenizerConfigURL, chatTemplateConfigURL: chatTemplateConfigURL)
        default:
            return LanguageModel(model: model, modelConfigURL: modelConfigURL, tokenizerDataURL: tokenizerDataURL, tokenizerConfigURL: tokenizerConfigURL, chatTemplateConfigURL: chatTemplateConfigURL)
        }
    }
}

extension LanguageModel {
    enum KVCacheAvailability {
        /// Language models that support KV cache via state. Implementation details for handling state
        /// encapsulated within the Core ML framework.
        ///
        /// Input: State
        /// Output: N/A
        case statefulKVCache
    }
}

@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModel {
    var metadata: [MLModelMetadataKey: Any] {
        model.modelDescription.metadata
    }

    var modelDescription: MLModelDescription {
        model.modelDescription
    }

    var description: String {
        if let description = metadata[MLModelMetadataKey.description] as? String,
           !description.isEmpty
        {
            return description
        }
        return model.configuration.modelDisplayName ?? ""
    }

    /// `name_or_path` in the Python world
    var modelName: String {
        if let userFields = metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String],
           let name = userFields["co.huggingface.exporters.name"]
        {
            return name
        }
        // This is usually the basename of the file, that's our best bet if no metadata exists
        guard let modelName = model.configuration.modelDisplayName else {
            fatalError("Models must have a name that identifies them")
        }
        return modelName
    }

    var inputIdsDescription: MLFeatureDescription {
        modelDescription.inputDescriptionsByName[Keys.inputIds]!
    }

    var inputIdsName: String {
        inputIdsDescription.name
    }

    /// The expected shape of the models latent sample input
    var inputIdsShape: [Int] {
        inputIdsDescription.multiArrayConstraint!.shape.map(\.intValue)
    }

    var isRequiringAttentionMask: Bool {
        modelDescription.inputDescriptionsByName[Keys.attentionMask] != nil
    }

    var isRequiringCausalMask: Bool {
        modelDescription.inputDescriptionsByName[Keys.causalMask] != nil
    }

    fileprivate static func kvCacheAvailability(for model: MLModel) -> KVCacheAvailability? {
        func isStatefulKVCacheAvailable(for model: MLModel) -> Bool {
            let kCacheState = model.modelDescription.stateDescriptionsByName[Keys.keyCache] != nil
            let vCacheState = model.modelDescription.stateDescriptionsByName[Keys.valueCache] != nil
            guard Set([kCacheState, vCacheState]).count == 1 else {
                fatalError("Invalid model configuration, expecting KV cache for states")
            }
            return kCacheState && kCacheState
        }

        func isDynamicallyShaped(_ description: MLFeatureDescription) -> Bool {
            guard let multiArrayConstraint = description.multiArrayConstraint else {
                return false
            }
            return switch multiArrayConstraint.shapeConstraint.type {
            case .unspecified: true
            case .enumerated: multiArrayConstraint.shapeConstraint.enumeratedShapes.count > 1
            case .range: true
            default: false
            }
        }

        if isStatefulKVCacheAvailable(for: model) {
            print("KVCache Vrai")
            return .statefulKVCache
        }
        let kCacheInput = model.modelDescription.inputDescriptionsByName[Keys.keyCache] != nil
        let vCacheInput = model.modelDescription.inputDescriptionsByName[Keys.valueCache] != nil
        let kCacheOutput = model.modelDescription.outputDescriptionsByName[Keys.presentKeys] != nil
        let vCacheOutput = model.modelDescription.outputDescriptionsByName[Keys.presentValues] != nil

        guard Set([kCacheInput, vCacheInput, kCacheOutput, vCacheOutput]).count == 1 else {
            fatalError("Invalid model configuration, expecting KV cache for inputs and outputs")
        }
        guard kCacheInput else {
            print("Pas de KVCache")
            return nil
        }
        // Check if cache is dynamic or not.
        let kCacheConstraint = model.modelDescription.inputDescriptionsByName[Keys.keyCache]!
        if isDynamicallyShaped(kCacheConstraint) {
            fatalError("""
                KV Cache using IO is currently not supported, please file a feature request on \
                https://github.com/huggingface/swift-transformers
                """)
        } else {
            fatalError("""
                KV Cache using IO is currently not supported, please file a feature request on \
                https://github.com/huggingface/swift-transformers
                """)
        }
    }
}

/// async properties downloaded from the configuration
@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModel {
    var modelConfig: Config {
        get {
            configuration!.modelConfig
        }
    }

    var tokenizerConfig: Config? {
        get {
            configuration!.tokenizerConfig
        }
    }

    var tokenizerData: Config {
        get {
            configuration!.tokenizerData
        }
    }

    var modelType: String? {
        get {
            modelConfig.modelType?.stringValue
        }
    }

    var textGenerationParameters: Config? {
        get {
            modelConfig.taskSpecificParams?.textGeneration
        }
    }

    var defaultDoSample: Bool {
        get {
            textGenerationParameters?.doSample?.boolValue ?? true
        }
    }

    var bosTokenId: Int? {
        get {
            let modelConfig = modelConfig
            return modelConfig.bosTokenId?.intValue
        }
    }

    var eosTokenId: Int? {
        get {
            let modelConfig = modelConfig
            return modelConfig.eosTokenId?.intValue
        }
    }

    var tokenizer: Tokenizer {
        get throws {
            if let _tokenizer {
                return _tokenizer
            }
            guard let tokenizerConfig = tokenizerConfig else {
                throw "Cannot retrieve Tokenizer configuration"
            }
            let tokenizerData = tokenizerData
            _tokenizer = try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
            return _tokenizer!
        }
    }
}

@available(macOS 15.0, iOS 18.0, *)
extension LanguageModel: TextGenerationModel {
    public var defaultGenerationConfig: GenerationConfig {
        var config = GenerationConfig(maxNewTokens: 2048)
        switch modelName.lowercased() {
        case let x where x.contains("gpt"):
            config.doSample = true
            config.topK = 50
        default: break
        }
        return config
    }
}


@available(macOS 15.0, iOS 18.0, *)
public class LanguageModelWithStatefulKVCache: LanguageModel {
    private enum Mode {
        case prefilling
        case extending
    }
    private var mode: Mode = .prefilling

    var state: MLState?

    public required init(
        model: MLModel,
        modelConfigURL: URL,
        tokenizerDataURL: URL,
        tokenizerConfigURL: URL? = nil,
        chatTemplateConfigURL: URL? = nil
    ) {
        super.init(model: model, modelConfigURL: modelConfigURL, tokenizerDataURL: tokenizerDataURL, tokenizerConfigURL: tokenizerConfigURL, chatTemplateConfigURL: chatTemplateConfigURL)
        // To support pre-filling and extend, the input must support
        // flexible shapes.
        guard maxContextLength - minContextLength > 1 else {
            fatalError("Expecting ranged query sequence length.")
        }
    }

    public override func resetState() async {
        state = model.makeState()
        mode = .prefilling
    }

    public override func predictNextTokenScores(
        _ tokens: MLTensor,
        config _: GenerationConfig
    ) async -> MLTensor {
        assert(tokens.rank == 2) // [batch, current sequence length]
        let tokenCount = tokens.shape[1]
        guard let state else {
            fatalError("""
                Encountered uninitialized `state`. Ensure `resetState` is called prior to calling \
                `predictNextTokenScores`. 
                """)
        }
        let inputIds = switch mode {
        case .prefilling: tokens // Pass in all takens if pre-filling prompt
        case .extending: tokens[nil, -1].expandingShape(at: 0) // otherwise just the last token
        }
        mode = .extending

        var inputDictionary = [
            Keys.inputIds: inputIds,
        ]
        if isRequiringAttentionMask {
            #if !((os(macOS) || (macCatalyst)) && arch(x86_64))
            // TODO: Infer scalar type from cache or model I/O descriptors
            let attentionMask = MLTensor(zeros: [1, 1, 1, tokenCount + 1], scalarType: Float16.self)
            inputDictionary[Keys.attentionMask] = attentionMask
            #else
            fatalError()
            #endif
        }
        if isRequiringCausalMask {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            // TODO: Infer scalar type from cache or model I/O descriptors
            let causalMask = MLTensor(zeros: [1, 1, 1, tokenCount + 1], scalarType: Float16.self)
            inputDictionary[Keys.causalMask] = causalMask
            #else
            fatalError()
            #endif
        }
        let outputs = try! await model.prediction(from: inputDictionary, using: state)

        assert(outputs.keys.contains(Keys.logits))
        let scores = outputs[Keys.logits]!
        assert(scores.rank == 3)
        let tokenIndex = inputIds.shape[1] - 1
        let nextTokenScores = scores[nil, tokenIndex, nil].expandingShape(at: 0)
        assert(nextTokenScores.rank == 3)
        assert(nextTokenScores.shape[0] == 1 && nextTokenScores.shape[1] == 1)
        return nextTokenScores
    }
}

extension String: @retroactive Error {}

