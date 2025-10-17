import fnmatch
from torch2grid.layer_visualizer import visualize_layers, create_layer_overview
from torch2grid.visualizer import visualize_grid
from torch2grid.transformer import to_neutral_grid


def list_layers(tensors):
    """Display all available layers with their shapes."""
    print("\nAvailable layers:")
    print("-" * 60)
    for idx, (name, arr) in enumerate(tensors.items(), 1):
        if arr is not None:
            shape_str = str(arr.shape) if hasattr(arr, 'shape') else "N/A"
            print(f"{idx:3d}. {name:40s} {shape_str}")
    print("-" * 60)


def filter_tensors_by_pattern(tensors, patterns):
    """
    Filter tensors by name patterns (supports wildcards).
    
    Args:
        tensors: Dictionary of layer names to arrays
        patterns: List of patterns (e.g., ['fc*', '*weight*'])
        
    Returns:
        Filtered dictionary of tensors
    """
    if not patterns:
        return tensors
    
    filtered = {}
    for pattern in patterns:
        for name, arr in tensors.items():
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                filtered[name] = arr
    return filtered


def filter_tensors_by_indices(tensors, indices):
    """
    Filter tensors by numeric indices.
    
    Args:
        tensors: Dictionary of layer names to arrays
        indices: List of 1-based indices
        
    Returns:
        Filtered dictionary of tensors
    """
    items = list(tensors.items())
    filtered = {}
    for idx in indices:
        if 1 <= idx <= len(items):
            name, arr = items[idx - 1]
            filtered[name] = arr
    return filtered


def parse_selection(selection_str, max_layers):
    """
    Parse user selection string into list of indices.
    Supports: "1", "1,3,5", "1-5", "1,3-5,7"
    
    Args:
        selection_str: User input string
        max_layers: Maximum number of layers
        
    Returns:
        List of indices
    """
    indices = []
    parts = selection_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                indices.append(int(part))
            except ValueError:
                continue
    
    return [i for i in indices if 1 <= i <= max_layers]


def interactive_mode(tensors, output_dir="grids"):
    """
    Interactive CLI for selecting and visualizing layers.
    
    Args:
        tensors: Dictionary of layer names to arrays
        output_dir: Base directory for saving visualizations
    """
    print("\n" + "="*60)
    print("torch2grid - Interactive Layer Selection")
    print("="*60)
    
    while True:
        list_layers(tensors)
        
        print("\nOptions:")
        print("  [numbers]    Select layers by number (e.g., '1,3,5' or '1-5')")
        print("  [pattern]    Filter by pattern (e.g., 'fc*', '*weight*')")
        print("  all          Visualize all layers")
        print("  overview     Create layer overview only")
        print("  unified      Create single unified grid")
        print("  list         Show layers again")
        print("  quit         Exit interactive mode")
        
        choice = input("\nYour choice: ").strip()
        
        if choice.lower() in ['quit', 'q', 'exit']:
            print("Exiting interactive mode.")
            break
        
        elif choice.lower() == 'list':
            continue
        
        elif choice.lower() == 'all':
            print("\nVisualizing all layers...")
            paths = visualize_layers(tensors, output_dir=f"{output_dir}/layers")
            overview = create_layer_overview(tensors, output_path=f"{output_dir}/layer_overview.png")
            print(f"\nCreated {len(paths)} layer visualizations + overview")
        
        elif choice.lower() == 'overview':
            print("\nCreating layer overview...")
            overview = create_layer_overview(tensors, output_path=f"{output_dir}/layer_overview.png")
            print(f"Created overview: {overview}")
        
        elif choice.lower() == 'unified':
            print("\nCreating unified grid...")
            grid = to_neutral_grid(tensors)
            path = visualize_grid(grid, output_dir=output_dir)
            print(f"Created unified grid: {path}")
        
        elif '*' in choice or '?' in choice:
            # Pattern matching
            patterns = [p.strip() for p in choice.split(',')]
            filtered = filter_tensors_by_pattern(tensors, patterns)
            
            if not filtered:
                print(f"No layers match pattern(s): {choice}")
                continue
            
            print(f"\nMatched {len(filtered)} layer(s):")
            for name in filtered.keys():
                print(f"  - {name}")
            
            confirm = input("\nVisualize these layers? (y/n): ").strip().lower()
            if confirm == 'y':
                paths = visualize_layers(filtered, output_dir=f"{output_dir}/layers")
                if len(filtered) > 1:
                    overview = create_layer_overview(filtered, output_path=f"{output_dir}/filtered_overview.png")
                print(f"\nCreated {len(paths)} visualizations")
        
        elif choice.replace(',', '').replace('-', '').replace(' ', '').isdigit():
            # Numeric selection
            indices = parse_selection(choice, len(tensors))
            
            if not indices:
                print("Invalid selection. Please try again.")
                continue
            
            filtered = filter_tensors_by_indices(tensors, indices)
            
            print(f"\nSelected {len(filtered)} layer(s):")
            for name in filtered.keys():
                print(f"  - {name}")
            
            confirm = input("\nVisualize these layers? (y/n): ").strip().lower()
            if confirm == 'y':
                paths = visualize_layers(filtered, output_dir=f"{output_dir}/layers")
                if len(filtered) > 1:
                    overview = create_layer_overview(filtered, output_path=f"{output_dir}/selected_overview.png")
                print(f"\nCreated {len(paths)} visualizations")
        
        else:
            print("Invalid option. Please try again.")
        
        print("\n" + "-"*60)
