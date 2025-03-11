//
//  File.swift
//  swift-transformers
//
//  Created by Thomas Le Bonnec on 10/03/2025.
//

import Foundation


// MARK: - Configuration files with dynamic lookup

@dynamicMemberLookup
public struct Config {
    public private(set) var dictionary: [NSString: Any]

    public init(_ dictionary: [NSString: Any]) {
        self.dictionary = dictionary
    }

    func camelCase(_ string: String) -> String {
        return string
            .split(separator: "_")
            .enumerated()
            .map { $0.offset == 0 ? $0.element.lowercased() : $0.element.capitalized }
            .joined()
    }
    
    func uncamelCase(_ string: String) -> String {
        let scalars = string.unicodeScalars
        var result = ""
        
        var previousCharacterIsLowercase = false
        for scalar in scalars {
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if previousCharacterIsLowercase {
                    result += "_"
                }
                let lowercaseChar = Character(scalar).lowercased()
                result += lowercaseChar
                previousCharacterIsLowercase = false
            } else {
                result += String(scalar)
                previousCharacterIsLowercase = true
            }
        }
        
        return result
    }


    public subscript(dynamicMember member: String) -> Config? {
        let key = (dictionary[member as NSString] != nil ? member : uncamelCase(member)) as NSString
        if let value = dictionary[key] as? [NSString: Any] {
            return Config(value)
        } else if let value = dictionary[key] {
            return Config(["value": value])
        }
        return nil
    }

    public var value: Any? {
        return dictionary["value"]
    }
    
    public var intValue: Int? { value as? Int }
    public var boolValue: Bool? { value as? Bool }
    public var stringValue: String? { value as? String }
    
    // Instead of doing this we could provide custom classes and decode to them
    public var arrayValue: [Config]? {
        guard let list = value as? [Any] else { return nil }
        return list.map { Config($0 as! [NSString : Any]) }
    }
    
    /// Tuple of token identifier and string value
    public var tokenValue: (UInt, String)? { value as? (UInt, String) }
}



extension Config {
    /// Assumes the file is already present at local url.
    /// `fileURL` is a complete local file path for the given model
    public static func loadConfig(fileURL: URL) throws -> Config {
        let data = try Data(contentsOf: fileURL)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw ConfigError.parse }
        return Config(dictionary)
    }
}



extension Config {
    enum ConfigError: Error {
        case parse
    }
}





public struct Configurations {
    public var modelConfig: Config
    public var tokenizerConfig: Config?
    public var tokenizerData: Config
    
    
    public static func loadConfigurations(
        modelConfigURL: URL,
        tokenizerDataURL: URL,
        tokenizerConfigURL: URL? = nil,
        chatTemplateConfigURL: URL? = nil
    ) throws -> Configurations {
        let defaultURL = URL(fileURLWithPath: "")
        // Load required configurations
        let modelConfig = try Config.loadConfig(fileURL: modelConfigURL)
        let tokenizerData = try Config.loadConfig(fileURL: tokenizerDataURL)
        // Load tokenizer config
        var tokenizerConfig = try? Config.loadConfig(fileURL: tokenizerConfigURL ?? defaultURL)
        // Check for chat template and merge if available
        if let chatTemplateConfig = try? Config.loadConfig(fileURL: chatTemplateConfigURL ?? defaultURL),
           let chatTemplate = chatTemplateConfig.chatTemplate?.stringValue {
            // The value of chat_template could also be an array of strings, but we're not handling that case here, since it's discouraged.
            // Create or update tokenizer config with chat template
            if var configDict = tokenizerConfig?.dictionary {
                configDict["chat_template"] = chatTemplate
                tokenizerConfig = Config(configDict)
            } else {
                tokenizerConfig = Config(["chat_template": chatTemplate])
            }
        }
        return Configurations(
            modelConfig: modelConfig,
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )
    }
}
