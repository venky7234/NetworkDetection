def extract_features(packet_dict):
    """Convert packet dict to feature list (dummy example)"""
    return [
        packet_dict.get("duration", 0),
        packet_dict.get("protocol_type", 0),
        packet_dict.get("service", 0),
        packet_dict.get("src_bytes", 0),
        packet_dict.get("dst_bytes", 0),
        packet_dict.get("flag", 0)
    ]