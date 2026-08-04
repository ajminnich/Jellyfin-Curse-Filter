using Jellyfin.Plugin.CurseFilter.Configuration;
using MediaBrowser.Common.Configuration;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Model.Serialization;

namespace Jellyfin.Plugin.CurseFilter;

/// <summary>Jellyfin profanity-filter preprocessing plugin.</summary>
public sealed class Plugin : BasePlugin<PluginConfiguration>
{
    /// <summary>Stable plugin identifier.</summary>
    public static readonly Guid PluginId = Guid.Parse("6f461ea8-0e47-4dc9-b27f-ab182516aed9");

    /// <summary>Initializes a new instance of the <see cref="Plugin"/> class.</summary>
    public Plugin(IApplicationPaths applicationPaths, IXmlSerializer xmlSerializer)
        : base(applicationPaths, xmlSerializer)
    {
        Instance = this;
    }

    /// <summary>Gets the current plugin instance.</summary>
    public static Plugin? Instance { get; private set; }

    /// <inheritdoc />
    public override string Name => "Curse Filter";

    /// <inheritdoc />
    public override string Description => "Makes caption- or audio-analyzed bleeped media the default while preserving each original as a reversible backup.";

    /// <inheritdoc />
    public override Guid Id => PluginId;
}
