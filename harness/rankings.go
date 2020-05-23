package harness

import (
	"fmt"
	"sort"
)

// MetricPair is a helper function for sorting the leaderboard
type MetricPair struct {
	c string
	m Metrics
}

// ByRank is a slice of metrics ranked for a leaderboard
type ByRank []MetricPair

func (m ByRank) Len() int      { return len(m) }
func (m ByRank) Swap(i, j int) { m[i], m[j] = m[j], m[i] }
func (m ByRank) Less(i, j int) bool {
	mi := m[i].m.Accuracy + m[i].m.Recall + m[i].m.Precision + m[i].m.F1
	mj := m[j].m.Accuracy + m[j].m.Recall + m[j].m.Precision + m[j].m.F1
	return mi > mj
}

// PrintResults printd out the results of the classifiers from best to worst
func PrintResults(results map[string]Metrics) {
	pl := make(ByRank, len(results))
	i := 0
	for k, v := range results {
		pl[i] = MetricPair{k, v}
		i++
	}
	sort.Sort(pl)

	for _, item := range pl {
		fmt.Printf("%-20s| %+v\n", item.c, item.m)
	}
}
