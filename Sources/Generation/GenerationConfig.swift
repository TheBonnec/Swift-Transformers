//
//  GenerationConfig.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

import Foundation

/// Essentials taken from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
public struct GenerationConfig {
    public var maxLength: Int
    public var maxNewTokens: Int
    public var doSample: Bool
    public var numBeams: Int
    public var numBeamGroups: Int
    public var penaltyAlpha: Double?
    public var temperature: Float
    public var topK: Int
    public var topP: Double
    public var repetitionPenalty: Double

    public var padTokenId: Int?
    public var bosTokenId: Int?
    public var eosTokenId: Int?

    public init(
        maxLength: Int = 200,
        maxNewTokens: Int,
        doSample: Bool = false,
        numBeams: Int = 1,
        numBeamGroups: Int = 1,
        penaltyAlpha: Double? = nil,
        temperature: Float = 0.7,
        topK: Int = 50,
        topP: Double = 1.0,
        repetitionPenalty: Double = 1.1
    ) {
        self.maxLength = maxLength
        self.maxNewTokens = maxNewTokens
        self.doSample = doSample
        self.numBeams = numBeams
        self.numBeamGroups = numBeamGroups
        self.penaltyAlpha = penaltyAlpha
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }
}

public extension GenerationConfig {
    var generationMode: GenerationMode {
        // Exclude this case from the pattern matching below
        // Inutile puis-ce qu'il conduit à une fatalerror lors de la génération 🙃
        /*
        if topK > 1, !doSample, penaltyAlpha != nil, penaltyAlpha! > 0 {
            return .contrastiveSearch
        }
        */

        switch (numBeams, numBeamGroups, doSample) {
        case (1, 1, false): return .greedy
        case (1, 1, true): return .sample
        case (2..., 1, false): return .beam
        case (2..., 2..., _): return .groupBeam
        default: return .unsupported
        }
    }
}

extension GenerationConfig: Decodable {}
